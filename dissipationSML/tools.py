import numpy as np
import gsw
import xarray as xr
from tqdm import tqdm
import regionmask as rm
from scipy.signal import convolve
from scipy.signal.windows import hann
from scipy.integrate import cumulative_trapezoid, trapezoid
import pandas as pd
from scipy.signal import butter, filtfilt
from dissipationSML import utilities

def add_densities(ds: xr.Dataset):
    """
    This function computes the potential density and its anomaly with respect to 0 dbar and 1000 dbar
    from the salinity and temperature data in the dataset. The computed values are added to the dataset if they do not
    already exist.

    Parameters
    ----------
    ds: xarray dataset containing the raw temperature and salinity data
    use_raw: bool
        If True, the function uses the raw temperature and salinity data to compute the potential density.
        If False, the function uses the corrected temperature and salinity data to compute the potential density

    Returns
    -------
    ds: xarray dataset with the additional variables SIG_THETA_RAW and SIGMA_T_RAW

    """
    vars = ['PSAL', 'TEMP']
    
    PSAL = ds[vars[0]].values
    TEMP = ds[vars[1]].values
    PRES = ds.PRES.values
    lon = ds.LONGITUDE
    lat = ds.LATITUDE
    CT = gsw.CT_from_t(PSAL, TEMP, PRES)  # Conservative temperature
    SA = gsw.SA_from_SP(PSAL,PRES, lon, lat)  # Absolute salinity
    SIGTHETA = gsw.pot_rho_t_exact(SA, TEMP, PRES, 0)  # Potential density
    SIGMA_T = gsw.density.sigma0(SA, CT)
    SIGMA_1 = gsw.density.sigma1(SA, CT)
    calculated = {'SIGTHETA': SIGTHETA, 'SIGMA_T': SIGMA_T, 'SIGMA_1': SIGMA_1}

    for var in calculated:
        ### add only if the variable is not already in the dataset
        var_name = var
        if var_name in ds:
            continue
        if var == 'SIGTHETA':
            long_name = 'potential density with respect to 0 dbar'
        elif var == 'SIGMA_T':
            long_name = 'potential density anomaly with respect to 0 dbar'
        elif var == 'SIGMA_1':
            long_name = 'potential density anomaly with respect to 1000 dbar'
        ds[var_name] = xr.DataArray(calculated[var], dims=('N_MEASUREMENTS'),
                               attrs={'units': 'kg/m^3', 'long_name': long_name})

    return ds

def add_vertical_water_velocity(ds: xr.Dataset, pitch_min, pitch_max) -> xr.Dataset:
    """
    Compute and add total vertical water velocity (W) to the dataset.

    W is estimated from the pressure rate of change and the glider vertical velocity model.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing 'DEPTH', 'TIME', and 'GLIDER_VERT_VELO_MODEL'.
    pitch_min : float
        Minimum pitch angle in degrees for climb profiles.
    pitch_max : float
        Maximum pitch angle in degrees for dive profiles.

    Returns
    -------
    ds : xarray.Dataset
        Dataset with added 'VERTICAL_WATER_VELOCITY' and 'VERTICAL_VELOCITY_MEASURED' variables.
    """
    # Extract variables
    time = ds['TIME'].values
    depth = ds['DEPTH'].values
    glider_velo = ds['W_MODEL'].values
    profile_number = ds['PROFILE_NUMBER'].values.astype(int)
    # Clean glider velocity data by using PITCH
    pitch = ds['PITCH'].values

    msk1 = ((profile_number % 2 == 0) & (pitch > pitch_min) & (pitch < pitch_max)) # Climb profiles with pitch in range 10 to 25
    msk2 = ((profile_number % 2 == 1) & (pitch > -pitch_max) & (pitch < -pitch_min)) # Dive profiles with pitch in range -25 to -10

    msk = msk1 | msk2

    glider_velo = xr.where(msk, glider_velo, np.nan)

    # Calculate vertical velocity from depth change (central difference, cm/s)
    ddepth = -(depth[2:] - depth[:-2]) * 100  # cm
    dtime = (time[2:] - time[:-2]) / np.timedelta64(1, 's')  # s

    # Handle invalid time intervals
    dtime[(dtime == 0) | (dtime > 500)] = np.nan

    # Estimate measured vertical velocity
    w_meas = ddepth / dtime
    w_meas = np.concatenate(([np.nan], w_meas, [np.nan]))  # Pad ends with NaN


    # Estimate vertical water velocity
    w_water = w_meas - glider_velo
    w_water[depth < 10] = np.nan  # Ignore shallow data
    ### ignore values above 10 cm/s
    w_water[np.abs(w_water) > 10] = np.nan  # Ignore extreme values

    # Create DataArrays with metadata
    da_w_meas = xr.DataArray(
        w_meas,
        dims=ds['W_MODEL'].dims,
        coords=ds['W_MODEL'].coords,
        name="W_MEASURED",
        attrs={
            "units": "cm/s",
            "description": "Measured vertical velocity from depth change"
        }
    )

    da_w_water = xr.DataArray(
        w_water,
        dims=ds['W_MODEL'].dims,
        coords=ds['W_MODEL'].coords,
        name="W_WATER",
        attrs={
            "units": "cm/s",
            "description": "Vertical water velocity (measured - glider model)"
        }
    )

    # Add results to dataset
    ds['W_MEASURED'] = da_w_meas
    ds['W_WATER'] = da_w_water

    ds.attrs['pitch_range'] = f"{pitch_min} to {pitch_max} degrees"

    return ds

def remove_spikes(ds, var, window=20, n_std=4, grad_threshold=1.0):
    """
    Remove spikes from a single variable in an xarray Dataset using
    a rolling mean / standard-deviation filter and a vertical gradient check.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing the variable to process. Must include a 'DEPTH' coordinate.
    var : str
        Name of the variable to clean (e.g., "TEMP", "PSAL").
    window : int, optional
        Size of the centered rolling window used to compute local mean and std. Default is 20.
    n_std : float, optional
        Number of standard deviations from the rolling mean used to identify spikes.
    grad_threshold : float, optional
        Maximum allowed absolute vertical gradient (dvar/dz). Values exceeding this
        threshold are marked as spikes. Default is 1.0.

    Returns
    -------
    xarray.Dataset
        The dataset with spike values replaced by NaN for the specified variable.
    """
    arr = ds[var].values
    depth = ds['DEPTH'].values

    # Compute gradient of variable with respect to depth
    grad_arr = ds[var].differentiate('DEPTH').values

    # Convert to pandas Series for rolling calculations
    s = pd.Series(arr)

    # Rolling mean and standard deviation
    roll_mean = s.rolling(window=window, center=True, min_periods=1).mean()
    roll_std = s.rolling(window=window, center=True, min_periods=1).std()

    deviation_mask = np.abs(s - roll_mean) > (n_std * roll_std)

    # ---------- Gradient-based spike detection (improved!) ----------
    # Identify only the true spike point:
    # where gradient is large, next gradient is large,
    # and gradients have opposite signs.
    grad_spike_mask = np.zeros_like(arr, dtype=bool)

    mask_pair = (
        (np.abs(grad_arr[:-1]) > grad_threshold) &
        (np.abs(grad_arr[1:]) > grad_threshold) &
        (np.sign(grad_arr[:-1]) != np.sign(grad_arr[1:]))
    )

    # The spike is at index i, not i+1
    grad_spike_mask[:-1] = mask_pair

    spike_mask = deviation_mask | grad_spike_mask

    # Replace spikes with NaN
    s[spike_mask] = np.nan

    # Update dataset
    ds[var].values = s.values

    return ds


def sorted_N2_profile(df, plev=20):
    """
    Compute adiabatic N² for a single vertical profile.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the profile data with columns 'PRES', 'TEMP', 'PSAL', 'LATITUDE', and 'LONGITUDE'.
    plev : float, optional
        Pressure level in dbar for averaging (default is 20 dbar)

    Returns
    -------
    n2_bray : 1D array
        Brunt–Väisälä frequency squared (N²) estimate
    """
    press = df['PRES'].to_numpy()
    temp = df['TEMP'].to_numpy()
    salinity = df['PSAL'].to_numpy()
    lat = float(np.nanmean(df['LATITUDE']))
    lon = float(np.nanmean(df['LONGITUDE']))

    SA = gsw.SA_from_SP(salinity, press, lon, lat)
    CT = gsw.CT_from_t(SA, temp, press)
    rho = gsw.rho(SA, CT, press)

    gravities = gsw.grav(lat, press)

    n = len(press)
    n2_bray = np.full(n, np.nan)
    p_bars = np.full(n, np.nan)
    alpha_1s = np.full(n, np.nan)

    for jj in range(n):
        if np.isnan(press[jj]):
            continue

        pmin_lev = max(press[jj] - plev / 2, np.nanmin(press))
        pmax_lev = min(press[jj] + plev / 2, np.nanmax(press))
        icyc = np.where((press >= pmin_lev) & (press <= pmax_lev))[0]

        if len(icyc) < 2:
            continue

        pbar = np.nanmean(press[icyc])
        p_bars[jj] = pbar

        pot_rho = gsw.pot_rho_t_exact(SA[icyc], temp[icyc], press[icyc], pbar)

        sv = 1 / pot_rho  
        press_pas = press[icyc] * 1e4  # dbar → Pa

        # Fast slope calculation (linear regression)
        x = press_pas
        y = sv
        alpha_1 = np.cov(x, y)[0, 1] / np.var(x)

        g = gravities[jj]
        rhobar = np.nanmean(rho[icyc])

        n2_bray[jj] = rhobar**2 * g**2 * -alpha_1
        alpha_1s[jj] = alpha_1

    ### interpolate n2_bray to match the original pressure levels
    ### if press is not increasing, sort and then interpolate and sort back
    if not np.all(np.diff(press) > 0):
        sorted_indices = np.argsort(press)
        press_sorted = press[sorted_indices]
        n2_bray_sorted = n2_bray[sorted_indices]
        p_bars_sorted = p_bars[sorted_indices]

        n2_bray_interp = np.interp(press_sorted, p_bars_sorted, n2_bray_sorted, left=np.nan, right=np.nan)

        # Sort back to original order
        n2_bray_interp = n2_bray_interp[np.argsort(sorted_indices)]
    else:
        n2_bray_interp = np.interp(press, p_bars, n2_bray, left=np.nan, right=np.nan)

    return pd.DataFrame({'SORTED_N2': n2_bray_interp, 'PRES':press, 'P_BAR': p_bars, "ALPHA_1": alpha_1s})

def add_adiabatic_sorted_N(ds, plev = 20):
    """
    Calculate adiabatic N² for each profile in the dataset and add it as a new variable.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset containing the necessary variables.
    plev : float, optional
        Pressure level in dbar for averaging (default is 20 dbar).
    
    Returns
    -------
    ds : xarray.Dataset
        The dataset with the adiabatic N² variable added.
    """
    groups = utilities.group_by_profiles(ds, ["PRES","TEMP","PSAL","LATITUDE","LONGITUDE", "DEPTH","TIME"])
    print(f"Calculating adiabatic N for {len(groups)} profiles...")
    df = groups.apply(sorted_N2_profile,plev=plev)

    dims = list(ds.sizes.keys())[0]
    
    SORTED_N_da = xr.DataArray(df['SORTED_N2'].to_numpy()**0.5, dims=dims, attrs={
        'long_name': 'Adiabatically sorted N',
        'units': '1/s',
        'plev': plev})

    ds['SORTED_N'] = SORTED_N_da

    return ds

def add_unsorted_N(ds, var_rho='SIGTHETA'):
    """
    Calculate adiabatic N² for each profile in the dataset and add it as a new variable.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset containing the necessary variables.
    
    Returns
    -------
    ds : xarray.Dataset
        The dataset with the adiabatic N² variable added.
    """
    profile_numbers = np.unique(ds.PROFILE_NUMBER.values)
    n_all = []
    for profile_number in tqdm(profile_numbers):
        profile_data = ds.where(ds.PROFILE_NUMBER == profile_number, drop=True)
        #PRES = profile_data.PRES.values
        #TEMP = profile_data.TEMP.values
        #PSAL = profile_data.PSAL.values
        z = profile_data.DEPTH.values
        rho = profile_data[var_rho].values
        n_profile = np.full_like(rho, np.nan, dtype=float)
        for i in range(2, len(z) - 2):
            dz = z[i + 2] - z[i - 2]
            drho = rho[i + 2] - rho[i - 2]
            if dz != 0 and not np.isnan(drho) and not np.isnan(dz) and not drho/dz < 0:
                n_profile[i] = np.sqrt((9.81 / 1027) * (drho / dz))
        n_all.append(n_profile)
    
    n_all = np.concatenate(n_all)
    dims = list(ds.sizes.keys())[0]
    n_all = xr.DataArray(n_all, dims=dims, attrs={
        'long_name': 'Unsorted Brunt-Väisälä frequency',
        'units': '1/s'
    })

    ds['N'] = n_all

    return ds

def add_velocity_scale(ds, var='W_WATER_HP', window_size_seconds=100):
    """
    Compute RMS of velocity in a moving window for each profile and add it to the dataset as 'VELOCITY_SCALE'.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with velocity and TIME coordinate.
    var : str
        Name of the velocity variable to process.
    window_size_seconds : float
        Size of the time window in seconds for computing RMS.
        
    Returns
    -------
    ds : xr.Dataset
        The dataset with an added 'VELOCITY_SCALE' variable.
    """
    profile_numbers = np.unique(ds.PROFILE_NUMBER.values)
    all_velocity_scales = []
    dims = list(ds.sizes.keys())[0]

    for profile_number in tqdm(profile_numbers, desc="Processing profiles"):
        if dims == 'N_MEASUREMENTS':
            ds_profile = ds.sel(N_MEASUREMENTS=ds.PROFILE_NUMBER == profile_number)
        elif dims == 'TIME':
            ds_profile = ds.sel(TIME=ds.PROFILE_NUMBER == profile_number)            

        time_vals = ds_profile.TIME.values
        if len(time_vals) < 2:
            # Not enough data for rolling window
            all_velocity_scales.append(xr.DataArray(np.full_like(ds_profile[var], np.nan)))
            continue

        # Calculate time deltas in seconds
        time_deltas = np.diff(time_vals) / np.timedelta64(1, 's')
        time_deltas = time_deltas[time_deltas < 100]  # Filter long gaps
        if len(time_deltas) == 0:
            mean_delta = 1  # Default to 1 second to avoid division by zero
        else:
            mean_delta = np.mean(time_deltas)

        # Convert time window to number of samples
        window_size = max(1, int(window_size_seconds / mean_delta))

        # Compute RMS
        velocity_squared = (ds_profile[var]/100) ** 2
        def rolling_rms(arr, window_size, dim="TIME"):
            rolled = arr.rolling({dim: window_size}, center=True)
            constructed = rolled.construct(f"{dim}_window")
            return constructed.mean(f"{dim}_window", skipna=True) ** 0.5
        #velocity_scale = velocity_squared.rolling(TIME=window_size, center=True).mean() ** 0.5
        velocity_scale = rolling_rms(velocity_squared, window_size, dim=dims)

        all_velocity_scales.append(velocity_scale)

    # Concatenate across profiles
    sigma_w = xr.concat(all_velocity_scales, dim=dims)
    #full_velocity_scale = full_velocity_scale.sortby(ds.TIME) / 100 # convert cm/s to m/s

    # Add to dataset
    ds['SIGMA_W'] = sigma_w
    ds['SIGMA_W'].attrs = {
        'long_name': 'Velocity variance scale',
        'description': f'RMS of {var} in ±{window_size_seconds / 2} second window per profile',
        'units': 'm/s',
        'window_size_seconds': window_size_seconds
    }
    ds['VELOCITY_SCALE'] = xr.DataArray(
        ds['SIGMA_W']**2,
        dims=dims,
        coords=ds['SIGMA_W'].coords,
        attrs={
            'long_name': 'Velocity scale sigma_w^2',
            'description': 'Squared velocity scale for further calculations',
            'units': 'm^2/s^2'
        }
    )

    return ds

def LEM_dissipation(ds, c=0.37):
    """
    Calculate turbulent kinetic energy dissipation rate using the Large Eddy Method (LEM):
        ε = c * N * (q')^2

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing 'ADIABATIC_N2' and 'VELOCITY_SCALE' variables.
    c : float, optional
        Model constant, default is 0.37 (typical value for LEM).

    Returns
    -------
    ds : xr.Dataset
        The dataset with a new variable 'DISSIPATION_LEM'.
    """
    # Ensure required variables exist
    if 'SORTED_N' not in ds or 'VELOCITY_SCALE' not in ds:
        raise ValueError("Dataset must contain 'SORTED_N2' and 'VELOCITY_SCALE' variables.")

    # Compute dissipation
    velocity_scale = ds['VELOCITY_SCALE']
    N = ds['SORTED_N']
    dissipation = c * N * velocity_scale
    
    ###mask all values below 1e-10
    #dissipation = dissipation.where(dissipation > 1e-10, drop=True)

    dissipation_log = np.log10(dissipation)

    # Add to dataset with metadata
    ds['EPSILON'] = dissipation
    ds['EPSILON'].attrs = {
        'long_name': 'Turbulent kinetic energy dissipation rate (LEM)',
        'description': "Calculated as ε = c * N * sigma_w² using Large Eddy Method",
        'units': 'W/kg',  # Assuming N in 1/s and q' in m/s
        'c_epsilon': c
    }

    ds.attrs['c_epsilon'] = c  # Store model constant in dataset attributes

    return ds

def find_boundary_OL(ds):
    profiles = np.unique(ds.PROFILE_NUMBER)
    boundary_OL = np.full(len(profiles), np.nan)
    
    for i, profile_i in enumerate(tqdm(profiles)):
        profile = ds.where(ds['PROFILE_NUMBER'] == profile_i, drop=True)
        temp = profile['TEMP'].values
        depth = profile['DEPTH'].values

        # remove NaNs
        isnan = ~np.isnan(temp) & ~np.isnan(depth)
        temp = temp[isnan]
        depth = depth[isnan]

        if len(temp) < 2:
            continue

        # sort by depth
        sort_idx = np.argsort(depth)
        depth = depth[sort_idx]
        temp = temp[sort_idx]

        # compute dT/dz
        dtemp_dz = np.gradient(temp, depth)

        # apply condition: T < 3 °C and |dT/dz| < 0.004
        mask = (temp < 3) & (np.abs(dtemp_dz) < 0.004)

        if np.any(mask):
            boundary_OL[i] = depth[mask].min()
    
    return boundary_OL


def find_boundary_AL(ds):
    profiles = np.unique(ds.PROFILE_NUMBER)
    boundary_AL = np.full(len(profiles), np.nan)
    
    for i, profile_i in enumerate(tqdm(profiles)):
        profile = ds.where(ds['PROFILE_NUMBER'] == profile_i, drop=True)
        temp = profile['TEMP'].values
        depth = profile['DEPTH'].values

        # remove NaNs
        isnan = ~np.isnan(temp) & ~np.isnan(depth)
        temp = temp[isnan]
        depth = depth[isnan]

        if len(temp) < 2:
            continue

        # sort by depth
        sort_idx = np.argsort(depth)
        depth = depth[sort_idx]
        temp = temp[sort_idx]

        # compute dT/dz
        dtemp_dz = np.gradient(temp, depth)

        # apply condition: T > 7.7 °C and |dT/dz| < 0.01
        mask = (temp > 7.7) & (np.abs(dtemp_dz) < 0.01) & (depth > 50)

        if np.any(mask):
            boundary_AL[i] = depth[mask].max()
    
    return boundary_AL

def avg_profiles(ds, var_name, binsize=5, HAB = False, log = False):
    """
    Averages profiles of a given variable over specified depth bins.
    If HAB is True, depth is calculated as HAB + BATHYMETRY, meaning height above bottom.
    """
    if HAB:
        BATH = ds.BATHYMETRY.values
        depth = -(ds.DEPTH.values + BATH)
    else:
        depth = ds.DEPTH.values
    profiles = ds.PROFILE_NUMBER
    var_values = ds[var_name].values
    if log:
        var_values = np.log10(var_values)

    var, depth_grid, profile_grid = utilities.construct_2dgrid(depth, profiles, var_values, xi=binsize, yi=np.max(profiles), agg='mean')
    return var, depth_grid, profile_grid

def mean_in_mld(ds: xr.Dataset, mld_ds: xr.Dataset, vars: list) -> xr.Dataset:
    """
    Calculate the mean of variables in the mixed layer depth (MLD) for each profile.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing variables (e.g., 'TEMP', 'PSAL', etc.) and 'DEPTH'.
    mld_ds : xarray.Dataset
        Dataset containing 'MLD' and 'PROFILE_NUMBER'.
    vars : list of str
        Variable names for which to compute the MLD mean.

    Returns
    -------
    xarray.Dataset
        The input MLD dataset with new variables added: '<var>_MEAN' for each input variable.
    """
    # Ensure we are working on a copy to avoid modifying in-place
    mld_ds = mld_ds.copy()

    dims = list(ds.sizes.keys())[0]
    for var in vars:
        print(f"Computing mean for {var} in MLD")
        mean_values = []

        for i in tqdm(range(len(mld_ds['PROFILE_NUMBER']))):
            profile_number = mld_ds['PROFILE_NUMBER'].values[i]
            mld_depth = mld_ds['MLD'].values[i]

            if np.isnan(mld_depth):
                mean_values.append(np.nan)
                continue

            # Extract variable and depth for current profile
            profile_mask = ds['PROFILE_NUMBER'] == profile_number
            profile_values = ds[var].where(profile_mask, drop=True)
            profile_depth = ds['DEPTH'].where(profile_mask, drop=True)

            # Mask to depths within MLD
            valid_mask = profile_depth <= mld_depth
            values_in_mld = profile_values.where(valid_mask, drop=True)

            # Compute mean and store
            mean_values.append(values_in_mld.mean().values)

        # Add the result to the MLD dataset
        mld_ds[var + '_MEAN'] = (dims, np.array(mean_values))

    return mld_ds

def integrate_in_mld(ds: xr.Dataset, mld_ds: xr.Dataset, vars: list, min_depth = None, max_depth = None) -> xr.Dataset:
    """
    Calculate the integral of variables in the mixed layer depth (MLD) for each profile.
    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing variables (e.g., 'TEMP', 'PSAL', etc.) and 'DEPTH'.
    mld_ds : xarray.Dataset
        Dataset containing 'MLD' and 'PROFILE_NUMBER'.
    vars : list of str
        Variable names for which to compute the MLD integral.
    min_depth : float or array-like
        Minimum depth to consider for integration. If an array, must match the length of profiles.
    max_depth : float or array-like
        Maximum depth to consider for integration. If an array, must match the length of profiles.
    Returns
    -------
    xarray.Dataset
        The input MLD dataset with new variables added: '<var>_TOTAL' and '<var>_MIN_DEPTH' for each input variable.
    """
    mld_ds = mld_ds.copy()

    for var in vars:
        print(f"Calculating integral for {var}...")
        integral_values = []
        min_valid_depth = []
        max_valid_depth = []
        profile_numbers = mld_ds['PROFILE_NUMBER'].values
        if min_depth is not None:
            if isinstance(min_depth, (int, float)):
                min_depth = np.full(len(profile_numbers), min_depth)
            elif len(min_depth) != len(profile_numbers):
                raise ValueError("min_depth must be a single value or match the number of profiles.")
        else:
            min_depth = np.zeros(len(profile_numbers))

        if max_depth is not None:
            if isinstance(max_depth, (int, float)):
                max_depth = np.full(len(profile_numbers), max_depth)
            elif len(max_depth) != len(profile_numbers):
                raise ValueError("max_depth must be a single value or match the number of profiles.")

        for i in tqdm(range(len(mld_ds['PROFILE_NUMBER']))):
            profile_number = mld_ds['PROFILE_NUMBER'].values[i]
            if max_depth is not None:
                mld_depth = max_depth[i]
            else:
                mld_depth = mld_ds['MLD'].values[i]

            if np.isnan(mld_depth):
                integral_values.append(np.nan)
                min_valid_depth.append(np.nan)
                max_valid_depth.append(np.nan)
                continue

            # Extract variable and depth for current profile
            profile_mask = ds['PROFILE_NUMBER'] == profile_number

            profile = ds.where(profile_mask, drop=True)
            # cut the profile to the MLD depth
            min_depth_value = min_depth[i]

            profile = profile.where((profile.DEPTH <= mld_depth) & (profile.DEPTH >= min_depth_value), drop=True)


            profile_var = profile[var].values
            profile_depth = profile['DEPTH'].values

            # Sort by depth
            sort_idx = np.argsort(profile_depth)
            depth_sorted = profile_depth[sort_idx]
            var_sorted = profile_var[sort_idx]

            # Exclude leading NaNs in values_sorted
            # Find first non-NaN index
            var_trimmed, start, end = trim_nan_edges(var_sorted)
            depth_trimmed = depth_sorted[start:end]
            if len(var_trimmed) == 0:
                # No valid data
                integral_values.append(np.nan)
                min_valid_depth.append(np.nan)
                max_valid_depth.append(np.nan)
                continue

            # Interpolate internal NaNs linearly
            msk = np.isnan(var_trimmed)
            
            if len(var_trimmed[~msk]) < 2:
                integral_values.append(np.nan)
                min_valid_depth.append(np.nan)
                max_valid_depth.append(np.nan)
                continue
            var_interp = np.interp(depth_trimmed, depth_trimmed[~msk], var_trimmed[~msk], left=np.nan, right=np.nan)

            integral_value = trapezoid(var_interp, depth_trimmed)
            integral_values.append(integral_value)
            min_valid_depth.append(np.min(depth_trimmed))
            max_valid_depth.append(np.max(depth_trimmed))

        mld_ds[var + '_TOTAL'] = ('TIME', integral_values)
        mld_ds[var + '_MIN_DEPTH'] = ('TIME', min_valid_depth)
        mld_ds[var + '_MAX_DEPTH'] = ('TIME', max_valid_depth)

    return mld_ds

def trim_nan_edges(arr):
    """
    Trims NaN values from the beginning and end of a 1D array.
    Parameters
    ----------
    arr: np.ndarray
        Input array to be trimmed.
    Returns
    -------
    trimmed_arr: np.ndarray
        Array with NaN values trimmed from the edges.
    first: int
        Index of the first non-NaN value.
    last: int
        Index of the last non-NaN value.
    """
    is_not_nan = ~np.isnan(arr)
    if not is_not_nan.any():
        return np.array([]), 0, 0
    first = np.argmax(is_not_nan)
    last = len(arr) - np.argmax(is_not_nan[::-1])
    return arr[first:last], first, last

def highpass_butterworth_time(ds, var, cutoff_period=330, order=4, max_interval=40):
    """
    Applies a highpass Butterworth filter to a variable over time, per profile.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset with TIME and PROFILE_NUMBER dimensions.
    var : str
        Variable to filter.
    cutoff_period : float, optional
        Highpass cutoff period in seconds (default 330s).
    order : int, optional
        Butterworth filter order (default 4).
    max_interval : float, optional
        Max gap (in seconds) to treat data as continuous (default 40s).

    Returns
    -------
    xr.Dataset
        Dataset with an added filtered variable: {var}_filtered.
    """
    def filter_profile(profile):
        time = profile.TIME.values
        dt = np.diff(time) / np.timedelta64(1, 's')
        dt[dt > max_interval] = np.nan
        if np.all(np.isnan(dt)):
            print("All time intervals are NaN, skipping profile.")
            return None
        mean_dt = np.nanmean(dt)
        if mean_dt == 0:
            print("Mean time interval is zero, skipping profile.")
            return None

        fs = 1 / mean_dt # 
        fc = 1 / cutoff_period ### critical frequency in Hz
        wn = 2 * fc /fs ## equals 2*mean_dt/cutoff_period
        b, a = butter(order, wn, btype='high')

        binned_df = utilities.bin_profile(profile, [var, 'DEPTH','PRES','TEMP','PSAL','LONGITUDE','LATITUDE','SIGMA_T','SIGTHETA'], binning=None, dim='TIME',max_interval = max_interval)
        signal = binned_df[var].values

        profile_filtered = np.full_like(signal, np.nan)
        trimmed, start, end = trim_nan_edges(signal)

        valid = ~np.isnan(trimmed)

        # Interpolate NaNs before filtering
        if np.isnan(trimmed).any():
            trimmed = pd.Series(trimmed).interpolate(
                method='linear', limit_direction='both').values

        if len(trimmed) > 3 * max(len(a), len(b)):
            filtered = filtfilt(b, a, trimmed)
            ### set all NaNs to NaN in the filtered signal
            filtered[~valid] = np.nan
            profile_filtered[start:end] = filtered

        binned_df[f'{var}_HP'] = profile_filtered
        binned_df['PROFILE_NUMBER'] = profile.PROFILE_NUMBER.values[0]
        return xr.Dataset.from_dataframe(binned_df.set_index('TIME'))

    # Process each profile with progress bar
    profile_numbers = np.unique(ds.PROFILE_NUMBER.values)
    filtered = [
        filter_profile(ds.where(ds.PROFILE_NUMBER==pn, drop=True))
        for pn in tqdm(profile_numbers, desc=f"Filtering {var}")
    ]

    filtered = [f for f in filtered if f is not None]
    if not filtered:
        return xr.Dataset()

    result = xr.concat(filtered, dim='TIME')
    result = result.sortby('TIME')
    result.attrs = ds.attrs
    result[f'{var}_HP'].attrs = ds[var].attrs
    result[f'{var}_HP'].attrs['filter'] = 'highpass_time'

    return result

def compute_mld(ds: xr.Dataset, variable, method: str = 'threshold', threshold = 0.03, ref_depth = 10,
                 use_bins: bool = True, binning: float = 10):
    """
    Computes the mixed layer depth (MLD) for each profile in the dataset. Two methods are available:
    1. **Threshold Method**: Computes MLD based on a density threshold (default is 0.03 kg/m³).
    2. **Convective Resistance (CR) Method**: Computes MLD based on the CR method. Values close to
    0 indicate a well-mixed layer, while values below 0 indicate a stratified layer. For the threshold,
    a value of -2 is recommended. (based on Frajka-Williams 2014, https://doi.org/10.1175/JPO-D-13-069.1)
    
    Parameters
    ----------
    ds : xr.Dataset
        The dataset containing the profiles.
    variable : str
        Variable used for the MLD calculation. For the CR method use density anomaly with reference to 1000 dbar ('SIGMA_1').
    method : str, optional
        The method to use for MLD calculation. Options are 'threshold' or 'CR'. Default is 'threshold'.
    threshold : float, optional
        If using 'threshold', this is the density threshold for MLD calculation. Default is 0.03 kg/m³.
        If using 'CR', this is the CR threshold for the convective resistance. A value of -2 is recommended.
    use_bins : bool, optional
        Whether to use binned data for MLD calculation. Default is True.
    binning : float, optional
        The binning resolution in meters. Default is 10m.

    Returns
    -------
    mld_values : pd.DataFrame
        A DataFrame containing the MLD values for each profile, along with the mean time for each profile.
        The DataFrame has columns: 'PROFILE_NUMBER', 'MLD' and 'TIME'.

    Notes
    -----
    Original Author: Till Moritz
    """
    if method == 'threshold':
        groups = utilities.group_by_profiles(ds, [variable, "DEPTH","TIME","LONGITUDE","LATITUDE"])
        mld = groups.apply(mld_profile_treshhold, variable=variable, threshold=threshold,
                            ref_depth=ref_depth, use_bins=use_bins, binning=binning)
    elif method == 'CR':
        if variable != 'SIGMA_1':
            print(f"Warning: {variable} can not be used for convective resistance calulation. Instead use SIGMA_1 for CR calculation.")
            variable = 'SIGMA_1'
        groups = utilities.group_by_profiles(ds, [variable, "DEPTH","TIME","LONGITUDE","LATITUDE"])
        if threshold > 0:
            print("Warning: CR threshold should be negative. Using -2 as default.")
            threshold = -2
        mld = groups.apply(mld_profile_CR, threshold=threshold, use_bins=use_bins, binning=binning)
    else:
        raise ValueError("Invalid MLD calculation method. Use 'threshold' or 'CR'.")
    # Convert the result to a DataFrame and name the MLD column
    mld = mld.reset_index(name='MLD')
    # Add mean time for each profile to the DataFrame
    mld['TIME'] = groups.TIME.mean().values
    mld['TIME'] = mld['TIME'].dt.round('1min')
    # Add longitude and latitude to the DataFrame
    mld['LONGITUDE'] = groups.LONGITUDE.mean().values
    mld['LATITUDE'] = groups.LATITUDE.mean().values
    return mld

def linear_interpolation(x, y, x_new):
    """Linearly interpolates y over x to estimate y at x_new."""
    return np.interp(x_new, x, y)

def mld_profile_treshhold(profile, variable: str = 'SIGMA_T', threshold: float = 0.03, ref_depth: float = 10,
                          use_bins: bool = False, binning: float = 10) -> float:
    """
    Computes the mixed layer depth (MLD) from a profile dataset based on the density profile, 
    using a threshold of 0.03 kg/m³

    Parameters
    ----------
    profile : pd.DataFrame or xr.Dataset
        Dataset or DataFrame containing depth and density columns.
    variable : str
        The name of the variable to use for the threshold calculation (default is 'SIGMA_T').
    threshold : float
        Density threshold for MLD estimation (default is 0.03 kg/m³).
    ref_depth : float
        Reference depth for MLD estimation (default is 10m).
    use_bins : bool
        Whether to bin the profile data before computing MLD.
    binning : float
        Bin resolution in meters if use_bins is True.

    Returns
    -------
    float
        Estimated mixed layer depth, or NaN if it cannot be determined.

    Notes
    -----
    Original Author: Till Moritz
    """
    
    if use_bins:
        profile = utilities.bin_profile(profile, [variable], binning=binning)
        depth = profile['DEPTH'].to_numpy()
        density = profile[variable].to_numpy()
    else:
        if isinstance(profile, pd.DataFrame):
            depth = profile['DEPTH'].to_numpy()
            density = profile[variable].to_numpy()
        elif isinstance(profile, xr.Dataset):
            depth = profile['DEPTH'].values
            density = profile[variable].values
        else:
            raise TypeError("Input must be a pandas.DataFrame or xarray.Dataset")
        
    # Convert to float arrays to avoid isfinite errors
    depth = np.asarray(depth, dtype=np.float64)
    density = np.asarray(density, dtype=np.float64)

    # Remove NaNs and check if valid data remains
    valid = np.isfinite(depth) & np.isfinite(density)
    if not np.any(valid):
        print("No valid depth or density data for MLD calculation.")
        return np.nan

    depth, density = depth[valid], density[valid]
    
    if np.nanmean(depth) < 0:
        depth = -1 * depth

    # Sort by depth
    sort_idx = np.argsort(depth)
    depth, density = depth[sort_idx], density[sort_idx]

    # check if any depth is below ref_depth
    if np.nanmin(depth) > ref_depth:
        print(f"No depth data below reference depth {ref_depth} m")
        return np.nan

    # Estimate density at reference depth
    if ref_depth in depth:
        idx_ref = np.nanargmin(np.abs(depth - ref_depth))
        density_ref = density[idx_ref]
    else:
        density_ref = linear_interpolation(depth, density, ref_depth)

    # Focus on depths below reference depth
    mask_below = depth > ref_depth
    depth_below = depth[mask_below]
    density_below = density[mask_below]

    if depth_below.size == 0:
        print(f"No data below reference depth {ref_depth} m")
        return np.nan

    if np.nanmax(density_below) < density_ref + threshold:
        print(f"No density values below reference depth {ref_depth} m exceed the threshold.")
        return np.nan

    # Find first crossing of the threshold
    for i in range(1, len(density_below)):
        if density_below[i] > density_ref + threshold:
            mld = (depth_below[i] + depth_below[i - 1]) / 2
            return round(mld, 1)

    return np.nan


def mld_profile_CR(profile, threshold: float = -2, use_bins: bool = True, binning: float = 10) -> float:
    """
    Calculate the mixed layer depth (MLD) using the Convective Resistance (CR) method.
    Returns NaN if no valid depth data below 10m is available or no CR values meet the threshold.

    Parameters
    ----------
    profile : xarray.Dataset or pandas.DataFrame
        Profile data containing 'DEPTH' and 'SIGMA_1'.
    threshold : float, optional
        CR threshold for determining MLD. Default is -2.
    use_bins : bool, optional
        Whether to apply depth binning. Default is False.
    binning : float, optional
        Bin size for depth binning, in meters. Default is 10.

    Returns
    -------
    float
        Computed MLD in meters, or NaN if criteria are not met.

    Notes
    -----
    Original author: Till Moritz
    """
    if use_bins:
        profile = utilities.bin_profile(profile, ['SIGMA_1'], binning=binning)

    CR_df = calculate_CR_for_all_depth(profile)
    depth = CR_df['DEPTH'].to_numpy()
    CR_values = CR_df['CR'].to_numpy()

    # Filter out NaNs
    valid = ~np.isnan(CR_values)
    depth = depth[valid]
    CR_values = CR_values[valid]

    if len(depth) == 0 or np.nanmin(depth) > 10:
        return np.nan

    # Identify where CR is below threshold
    below_threshold = CR_values < threshold
    if not np.any(below_threshold):
        return np.nan

    mld =  np.nanmin(depth[below_threshold])
    return round(mld, 1)


def calculate_CR_for_all_depth(profile):
    """
    Calculate CR for all depths in the profile.

    Parameters
    ----------
    profile : xarray.Dataset or pandas.DataFrame
        Profile data containing depth and SIGMA_1.
    use_bins : bool, optional
        If True, bins the data before computation. Default is False.
    binning : float, optional
        Bin size for binning. Default is 10.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: DEPTH, CR.

    Notes
    -----
    Original Author: Till Moritz
    """
    required_vars = ['DEPTH', 'SIGMA_1']

    if isinstance(profile, xr.Dataset):
        df_profile = profile[required_vars].to_dataframe().reset_index()
    elif isinstance(profile, pd.DataFrame):
        df_profile = profile[required_vars].copy()
    else:
        raise TypeError("Input must be an xarray.Dataset or pandas.DataFrame.")

    # Drop NaNs in SIGMA_1 to avoid problems
    df_profile = df_profile.dropna(subset=['SIGMA_1'])

    # Prepare CR values
    CR_values = []
    depths = df_profile['DEPTH'].to_numpy()

    for h in depths:
        CR_h = compute_CR(df_profile, h)
        CR_values.append(CR_h)
    
    #return CR_values, depths
    return pd.DataFrame({'DEPTH': depths, 'CR': CR_values})

def compute_CR(profile, h: float) -> float:
    """
    Compute the CR (density anomaly integral) up to the reference depth h.

    Parameters
    ----------
    profile : xarray.Dataset or pandas.DataFrame
        Profile data containing 'DEPTH' and 'SIGMA_1'.
    h : float
        Reference depth up to which CR is computed.

    Returns
    -------
    float
        Computed CR up to the depth h, or NaN if insufficient data.

    Notes
    -----
    Original Author: Till Moritz
    """
    # Extract depth and sigma_1 depending on input type
    if isinstance(profile, xr.Dataset):
        depth = profile['DEPTH'].values
        sigma1 = profile['SIGMA_1'].values
    elif isinstance(profile, pd.DataFrame):
        depth = profile['DEPTH'].to_numpy()
        sigma1 = profile['SIGMA_1'].to_numpy()
    else:
        raise TypeError("Input must be an xarray.Dataset or pandas.DataFrame.")

    # Filter out NaNs
    valid = ~np.isnan(depth) & ~np.isnan(sigma1)
    depth = depth[valid]
    sigma1 = sigma1[valid]

    if len(depth) < 2 or h > np.nanmax(depth):
        return np.nan

    # Sort by depth
    idx = np.argsort(depth)
    depth = depth[idx]
    sigma1 = sigma1[idx]

    # Select depths up to h
    mask = (depth <= h) & (depth >= 0)
    if np.sum(mask) < 1:
        print(f"Not enough data points for depth {h} m")
        return np.nan

    depth_masked = depth[mask]
    sigma1_masked = sigma1[mask]

    # Fill in missing top layer if needed
    min_depth = depth_masked[0]
    
    if min_depth > 0 and min_depth <= 15:
        new_depth = np.arange(0, min_depth, 0.25)
        top_mask = depth_masked <= 15
        sigma1_top_mean = np.nanmean(sigma1_masked[top_mask])
        new_sigma1 = np.full_like(new_depth, sigma1_top_mean, dtype=float)

        depth_filled = np.concatenate([new_depth, depth_masked])
        sigma1_filled = np.concatenate([new_sigma1, sigma1_masked])
    else:
        depth_filled = depth_masked
        sigma1_filled = sigma1_masked
    # Integration and CR computation
    integral = cumulative_trapezoid(sigma1_filled, depth_filled, initial=0)[-1]
    sigma1_h = sigma1_filled[-1]
    CR_h = integral - np.nanmax(depth_filled) * sigma1_h

    return CR_h

def min_max_depth_per_profile(ds: xr.Dataset):
    """
    This function computes the maximum depth for each profile in the dataset

    Parameters
    ----------
    ds: xarray on OG1 format containing at least depth and profile_number. Data
    should not be gridded.

    Returns
    -------
    max_depths: pandas dataframe containing the profile number and the maximum depth of that profile
    """
    max_depths = ds.groupby('PROFILE_NUMBER').apply(lambda x: x['DEPTH'].max())
    min_depths = ds.groupby('PROFILE_NUMBER').apply(lambda x: x['DEPTH'].min())
    ### add the unit to the dataarray
    max_depths.attrs['units'] = ds['DEPTH'].attrs['units']
    min_depths.attrs['units'] = ds['DEPTH'].attrs['units']
    return min_depths, max_depths

def cut_region(ds: xr.Dataset,region: rm.Regions):
    """
    This function cuts the dataset to the region specified by the regionmask region

    Parameters
    ----------
    ds: xarray dataset containing the data
    region: regionmask region object specifying the region to cut to

    Returns
    -------
    ds_region: xarray dataset containing the data only in the specified region
    """
    if "longitude" in ds.coords:
        region_mask = region.mask(ds.longitude,ds.latitude)

    else:
        region_mask = region.mask(ds.LONGITUDE, ds.LATITUDE)
        ds_region = ds.isel(TIME=region_mask == 0)

    return ds_region


def match_era5_to_mld(ds_mld, ds_era5, lon_range=None, lat_range=None, time_lag = None, time_range = None):
    """
    Match ERA5 data to MLD observations based on time, longitude, and latitude. 
    ERA5 values are averaged over a spatial range (if provided) and matched by nearest time.

    Parameters
    ----------
    ds_mld : xarray.Dataset
        Dataset with MLD observations and variables: TIME, LATITUDE, LONGITUDE.
    ds_era5 : xarray.Dataset
        ERA5 dataset with dimensions: valid_time, latitude, longitude.
    lon_range : float or None, optional
        Longitude range (in degrees) for spatial averaging. If None, uses nearest longitude.
    lat_range : float or None, optional
        Latitude range (in degrees) for spatial averaging. If None, uses nearest latitude.
    time_lag: float
        Take the profile time minus a time_lag [hours] for the matching of the ERA5 data.
    time_range: float
        Takes a time range [hours], in which the ERA5 data is averaged.

    Returns
    -------
    xarray.Dataset
        MLD dataset with ERA5 variables added as 1D arrays aligned with TIME.
    """
    times = ds_mld.TIME.values
    lons = ds_mld.LONGITUDE.values
    lats = ds_mld.LATITUDE.values

    if time_lag:
        times = times - np.timedelta64(time_lag,"h")

    matched_profiles = []

    for i in tqdm(range(len(times)), desc="Matching ERA5 to MLD"):
        time = times[i]
        lon = lons[i]
        lat = lats[i]

        if time_range:
            match = ds_era5.sel(valid_time=slice(time, time+np.timedelta64(time_range,'h')))
            match = match.mean(dim=["valid_time"], skipna=True, keep_attrs=True)
        else:
            # Select nearest ERA5 time
            match = ds_era5.sel(valid_time=time, method="nearest")

        # Apply spatial window if specified
        if lon_range:
            match = match.sel(longitude=slice(lon - lon_range, lon + lon_range))
        else:
            match = match.sel(longitude=slice(lon))

        if lat_range:
            match = match.sel(latitude=slice(lat - lat_range, lat + lat_range))
        else:
            match = match.sel(latitude=slice(lat))

        # Compute spatial mean and assign timestamp
        match = match.mean(dim=["latitude", "longitude"], skipna=True, keep_attrs=True)
        match = match.assign_coords(TIME=time)
        matched_profiles.append(match)

    # Combine matched ERA5 profiles into one dataset
    ds_matched = xr.concat(matched_profiles, dim="TIME")

    # Merge new ERA5 variables into the MLD dataset
    for var in ds_era5.data_vars:
        if var not in ds_mld:
            ds_mld[var] = ds_matched[var]

    # Store matching settings as metadata
    ds_mld.attrs["longitude_range_used"] = f"±{lon_range}°" if lon_range else "Nearest longitude"
    ds_mld.attrs["latitude_range_used"] = f"±{lat_range}°" if lat_range else "Nearest latitude"
    ds_mld.attrs["time_lag"] = f"{time_lag} hours" if time_lag else "Nearest point in time"

    return ds_mld


def add_buoyancy_flux(ds:xr.Dataset, c_p=4e3, g=9.81, L=2.5e6):
    """
    Add surface buoyancy flux to the dataset based on the surface heat fluxes.

    Parameters
    ----------
    ds : xarray.Dataset or pandas.DataFrame
        Dataset of Dataframe containing temperature and salinity.
    c_p : float, optional
        Specific heat capacity of seawater (default is 4e3 J/(kg*K)).
    g : float, optional
        Gravitational acceleration (default is 9.81 m/s^2).
    L : float, optional
        Latent heat of vaporization (default is 2.5e6 J/kg).
    
    Returns
    -------
    xarray.Dataset
        Dataset with buoyancy flux added.
    """
    Q_SW = ds['ssr'] # net shortwave radiation in W/m2
    Q_LW = ds['str'] # net longwave radiation in W/m2
    Q_LH = ds['slhf'] # net latent heat flux in W/m2
    Q_SH = ds['sshf'] # net sensible heat flux in W/m2
    Q_0 = Q_SW + Q_LW + Q_LH + Q_SH # net surface heat flux
    
    E = ds['e'] # evaporation
    P = ds['tp'] # precipitation

    S = ds['PSAL_MEAN'] # Salinity in the surface miyed layer
    T = ds['TEMP_MEAN'] # Temperature in the surface mixed layer
    rho = ds['SIGTHETA_MEAN'] # Density in the surface mixed layer

    alpha = gsw.alpha(S, T, 0) # Thermal expansion coefficient
    beta = gsw.beta(S, T, 0) # Haline expansion coefficient

    B_0 = -g/rho * (alpha/c_p * Q_0 + beta * Q_LH/L * S) # buoyancy flux like Evans

    ds['B_0'] = B_0
    return ds

def dissipation_bouyancy_flux(ds, MLD = None):
    """
    Add surface buoyancy flux to the dataset based on the surface heat fluxes.

    Parameters
    ----------
    ds : xarray.Dataset or pandas.DataFrame
        Dataset of Dataframe containing temperature and salinity.
    c_p : float, optional
        Specific heat capacity of seawater (default is 4e3 J/(kg*K)).
    g : float, optional
        Gravitational acceleration (default is 9.81 m/s^2).
    L : float, optional
        Latent heat of vaporization (default is 2.5e6 J/kg).
    MLD : array-like, optional
        Mixed layer depth to use for dissipation calculation. If None, uses 'MLD' from ds.
    
    Returns
    -------
    xarray.Dataset
        Dataset with buoyancy flux added.
    """
    
    ds = add_buoyancy_flux(ds)
    
    # Calculate dissipation rate based on buoyancy flux for all positive values of B_0
    B_0 = ds['B_0']
    #B_0 = B_0.where(B_0 > 0 , np.nan)  # Set negative values to zero
    if MLD is None:
        MLD = ds['MLD']  # Mixed layer depth [m]
    ds['EPSILON_Q'] = 1/2 * MLD * B_0
    
    return ds

def add_u_star(ds):
    """
    Add the friction velocity u_star to the dataset based on the wind stress and density.
    
    Parameters
    ----------
    ds : xarray.Dataset or pandas.DataFrame
        Dataset of Dataframe containing wind stress and density.
    
    Returns
    -------
    xarray.Dataset
        Dataset with u_star added.
    """
    
    tau = ds['TAU']  # Wind stress [N/m^2]
    rho = ds['SIGTHETA_MEAN']  # Density [kg/m^3]
    
    # Calculate friction velocity
    u_star = np.sqrt(tau / rho)
    
    ds['U_STAR'] = u_star
    return ds

def add_hs(ds):
    """
    Add the transition depth (Stokes depth) hs to the dataset. The calculation is based on Buckingham 2019.

    Parameters
    ----------
    ds : xarray.Dataset or pandas.DataFrame
        Dataset of Dataframe containing significant wave height and wind stress.
    Returns
    -------
    xarray.Dataset
        Dataset with transition depth hs and Stokes depth H_S added.
    """
    g = 9.81 # gravitational acceleration [m/s^2]
    kappa = 0.4 # von Karman constant 
    T = ds['pp1d'] # peak wave period [s]
    u_star_wind = ds['zust'] # wind friction velocity [m/s]
    c_p = g*T/(2*np.pi) # phase speed of peak wave [m/s]
    #c_bar = 0.1*c_p # effective wave speed [m/s], based on Buckingham 2019

    swh = ds['swh'] # significant wave height [m]
    add_u_star(ds)  # Ensure u_star is calculated and added to the dataset
    #u_star = ds['U_STAR'] # friction velocity from wind stress (ERA-5) [m/s]‚

    # Calculate the transition depth
    h_s = 0.38 * swh * c_p/u_star_wind

    ds['H_S'] = h_s

    return ds

def dissipation_wind_stress(ds):
    """
    Add wind stress to the dataset based on the wind speed and friction velocity.

    Parameters
    ----------
    ds : xarray.Dataset or pandas.DataFrame
        Dataset of Dataframe containing temperature and salinity.
    
    Returns
    -------
    xarray.Dataset
        Dataset with wind stress added.
    """
    
    #ds = add_wind_stress(ds)
    #ds = add_hs(ds)
    
    # Extract variables from dataset
    MLD = ds['MLD']             # Mixed layer depth [m]
    tau = ds['TAU']             # Wind stress [N/m^2]
    rho = ds['SIGTHETA_MEAN']   # Density [kg/m^3]
    min_depth = ds['DISSIPATION_LEM_MIN_DEPTH']  # Minimum depth for dissipation calculation [m]
    max_depth = ds['DISSIPATION_LEM_MAX_DEPTH']  # Maximum depth for dissipation calculation [m]
    #print(min_depth)

    # Physical constant
    kappa = 0.4  # Von Karman constant
    U_STAR = np.sqrt(tau/rho)
    # Compute epsilon_tau using conditional logic
   
    EPSILON_TAU = xr.where(min_depth > max_depth , np.nan , 
                           (U_STAR ** 3) / kappa * np.log(max_depth / min_depth))
    ds['EPSILON_TAU'] = EPSILON_TAU
    
    return ds

def get_background_dissipation(ds, profile_range=[0, -1], depth_range=[200, 400], sg005_ds=None):
    """
    Calculate the background dissipation from the dataset and optionally compare with sg005_ds.
    
    Parameters:
    - ds: xarray Dataset
    - profile_range: List [start, end] of profile numbers
    - depth_range: List [min_depth, max_depth]
    - sg005_ds: Optional xarray Dataset to compare
    
    Returns:
    - Tuple: (background_sg005, ds_background) if sg005_ds provided, else (None, ds_background)
    """
    
    def cut_background(ds, profile_range, depth_range):
        return ds.where(
            (ds.PROFILE_NUMBER >= profile_range[0]) &
            (ds.PROFILE_NUMBER <= profile_range[1]) &
            (ds.DEPTH >= depth_range[0]) &
            (ds.DEPTH <= depth_range[1]),
            drop=True
        )

    def plot_hist(ax, ds, median=None, mean=None, color='blue'):
        figh, axh = plotting.plot_histogram(
            ds, vars=['DISSIPATION_LEM'], bins=50, log_scale=True, density=True,
            alpha=0.2, color=color, edgecolor='black', ax=ax
        )
        unit = utilities.get_unit(ds, 'DISSIPATION_LEM')
        if mean is not None:
            ax.axvline(np.log10(mean), color=color, linestyle='--', label=f'Mean = {mean:.2e} [{unit}]')
        if median is not None:
            ax.axvline(np.log10(median), color=color, linestyle=':', label=f'Median = {median:.2e} [{unit}]')
        ax.set_title(f'Background Dissipation Histogram of sg{ds.Glider +'/'+ ds.Mission} (c = {ds.c_epsilon})')
        ax.legend()
        return ax

    ds_background = cut_background(ds, profile_range, depth_range)
    mean_bg = np.nanmean(ds_background['DISSIPATION_LEM'].values)
    median_bg = np.nanmedian(ds_background['DISSIPATION_LEM'].values)

    background_sg005 = None

    mean_ratio, median_ratio, ratio_total = None, None, None

    if sg005_ds is None:
        # Just one histogram
        fig, ax = plt.subplots(figsize=(8, 5))
        plot_hist(ax, ds_background, median_bg, mean_bg, color='blue', label='Background')
    else:
        # Load ranges and cut sg005 dataset
        profile_range_005 = bg_yaml['sg005']['005/20080606']['profile_range']
        depth_range_005 = bg_yaml['sg005']['005/20080606']['depth_range']
        background_sg005 = cut_background(sg005_ds, profile_range_005, depth_range_005)

        mean_005 = np.nanmean(background_sg005['DISSIPATION_LEM'].values)
        median_005 = np.nanmedian(background_sg005['DISSIPATION_LEM'].values)

        # Create combined figure with 3 subplots
        fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

        # Subplot 1: original dataset
        axs[0].set_title(f'Background Dissipation Histogram of {ds.id.split("T")[0]} (c = {ds.c_epsilon})')
        plot_hist(axs[0], ds_background, median_bg, mean_bg, color='blue')

        # Subplot 2: sg005 dataset
        axs[1].set_title(f'Background Dissipation Histogram of {sg005_ds.id.split("T")[0]} (c = {sg005_ds.c_epsilon})')
        plot_hist(axs[1], background_sg005, median_005, mean_005, color='red')

        # Subplot 3: comparison
        plotting.plot_histogram(ds_background, vars=['DISSIPATION_LEM'], bins=50, log_scale=True, density=True,
                                 alpha=0.2, color='blue', edgecolor='darkblue', ax=axs[2])
        plotting.plot_histogram(background_sg005, vars=['DISSIPATION_LEM'], bins=50, log_scale=True, density=True,
                                 alpha=0.2, color='red', edgecolor='darkred', ax=axs[2])
        axs[2].set_title('Comparison of Background Dissipation Histograms')
        axs[2].legend(['sg' + ds_background.Glider + '/' + ds_background.Mission, 'sg005/20080606'])

        c_005 = bg_yaml['sg005']['005/20080606']['c']
        mean_ratio = mean_005 / mean_bg
        c_mean = c_005 * mean_ratio
        median_ratio = median_005 / median_bg
        c_median = c_005 * median_ratio
        ratio_total = (mean_ratio + median_ratio) / 2
        c_total = c_005 * ratio_total

        print(f'Mean ratio (sg005/Background): {mean_ratio:.3f}, c = {c_mean:.3f}')
        print(f'Median ratio (sg005/Background): {median_ratio:.3f}, c = {c_median:.3f}')
        print(f'Average ratio (sg005/Background): {ratio_total:.3f}, c = {c_total:.3f}')

    # Show section plot
    figs, section_axs = plotting.plot_section(ds_background, vars=['SORTED_N2', 'VELOCITY_SCALE_2_LOG', 'DISSIPATION_LEM_LOG'], v_res=2)
    plt.show()

    return mean_ratio, median_ratio, ratio_total
