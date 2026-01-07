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

def remove_spikes(ds, vars = ["TEMP","PSAL"], window=20, n_std=0.03, grad_th = [0.5,1]):
    """
    Removing spikes from a variable in an xarray Dataset using a rolling mean filter.
    Parameters
    ----------
    ds: xarray Dataset containing the variable to be processed
    var: str, name of the variable to process
    window: int, size of the rolling window
    threshold: float, threshold for spike detection
    
    Returns
    -------
    ds: xarray Dataset with spikes removed from the specified variable
    """
    if isinstance(vars, str):
        vars = list([vars])
    for n, var in enumerate(vars):
        arr = ds[var].values
        depth = ds.DEPTH.values
        grad_arr = np.gradient(arr, depth)
        s = pd.Series(arr)

        # Compute rolling mean excluding the current point (centered)
        roll_mean = s.rolling(window=window, center=True, min_periods=1).mean()
        roll_std = s.rolling(window=window, center=True, min_periods=1).std()

        # Mark points that n_std from the local mean
        # and mark points that are outside of the gradient threshold
        mask = (np.abs(s - roll_mean) > (n_std * roll_std)) | (np.abs(grad_arr) > grad_th[n])

        s[mask] = np.nan
        ds[var].values = s.values
    return ds

def add_SIG1_CT_SA(ds: xr.Dataset):
    """
    This function computes the potential density and its anomaly with respect to 0 dbar and 1000 dbar
    from the salinity and temperature data in the dataset. The computed values are added to the dataset if they do not
    already exist.

    Parameters
    ----------
    ds: xarray dataset containing the raw temperature and salinity data

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
    SIGMA_1 = gsw.density.sigma1(SA, CT)
    calculated = {'SIGMA_1': SIGMA_1, 'SA': SA, 'CT': CT}
    long_names = {'SA': 'Absolute Salinity', 'CT': 'Conservative Temperature',
                  'SIGMA_1': 'potential density anomaly with respect to 1000 dbar'}

    for var in calculated:
        ds[var] = xr.DataArray(calculated[var], dims=('N_MEASUREMENTS'),
                               attrs={'units': 'kg/m^3', 'long_name': long_names[var]})
        
    return ds

def add_vertical_water_velocity(ds: xr.Dataset, pitch_min, pitch_max) -> xr.Dataset:
    """
    Compute and add total vertical water velocity (W) to the dataset.

    W is estimated from the pressure rate of change and the glider vertical velocity model.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing 'DEPTH', 'TIME', and 'W_M'.
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
    glider_velo = ds['W_M'].values
    profile_number = ds['PROFILE_NUMBER'].values.astype(int)
    # Clean glider velocity data by using PITCH
    pitch = ds['PITCH'].values
    print(profile_number.dtype)

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
        dims=ds['W_M'].dims,
        coords=ds['W_M'].coords,
        name="W_MEAS",
        attrs={
            "units": "cm/s",
            "description": "Measured vertical velocity from depth change"
        }
    )

    da_w_water = xr.DataArray(
        w_water,
        dims=ds['W_M'].dims,
        coords=ds['W_M'].coords,
        name="W_W",
        attrs={
            "units": "cm/s",
            "description": "Vertical water velocity (measured - glider model)"
        }
    )

    # Add results to dataset
    ds['W_MEAS'] = da_w_meas
    ds['W_W'] = da_w_water

    ds.attrs['pitch_range'] = f"{pitch_min} to {pitch_max} degrees"

    return ds

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

        fs = 1 / mean_dt # sampling frequency in Hz
        fc = 1 / cutoff_period ### cutoff frequency in Hz
        wn = 2 * fc /fs ## equals 2*mean_dt/cutoff_period
        b, a = butter(order, wn, btype='high')

        vars_to_bin = list(profile.data_vars.keys()) + ['DEPTH','LONGITUDE','LATITUDE']
        binned_df = utilities.bin_profile(profile, vars=vars_to_bin, binning=None, dim='TIME',max_interval = max_interval)
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

def highpass_butterworth_depth(ds, var, cutoff=30, order=4, max_interval=5):
    """
    Applies a highpass Butterworth filter to a variable over DEPTH, per profile.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset with DEPTH and PROFILE_NUMBER dimensions.
    var : str
        Variable to filter.
    cutoff_depth : float, optional
        Highpass cutoff wavelength in meters (default 30 m).
    order : int, optional
        Butterworth filter order (default 4).
    max_interval : float, optional
        Max gap (in meters) to treat data as continuous (default 5 m).

    Returns
    -------
    xr.Dataset
        Dataset with an added filtered variable: {var}_HP.
    """
    def filter_profile(profile):
        depth = profile.DEPTH.values
        ### sort profile by depth
        sort_idx = np.argsort(depth)
        dz = np.diff(depth[sort_idx])
        dz[dz > max_interval] = np.nan

        if np.all(np.isnan(dz)):
            print(f"All depth intervals are NaN, skipping profile {profile.PROFILE_NUMBER.values[0]}")
            return None

        mean_dz = np.nanmean(dz)
        if mean_dz == 0 or np.isnan(mean_dz):
            print(f"Mean depth interval invalid, skipping profile {profile.PROFILE_NUMBER.values[0]}")
            return None

        # Convert cutoff depth (in m) to a spatial frequency (cycles per meter)
        fs = 1 / mean_dz  # "sampling frequency" in samples per meter
        fc = 1 / cutoff  # cutoff frequency in cycles per meter
        wn = 2 * fc / fs  # normalized critical frequency (Nyquist = 1)

        if wn >= 1:
            print(f"Cutoff frequency too high for profile {profile.PROFILE_NUMBER.values[0]}, skipping.")
            return None

        b, a = butter(order, wn, btype='high')

        vars_to_bin = list(profile.data_vars.keys()) + ['DEPTH', 'LONGITUDE', 'LATITUDE']
        binned_df = utilities.bin_profile(profile, vars=vars_to_bin, binning=None, dim='DEPTH', max_interval=max_interval)

        signal = binned_df[var].values
        profile_filtered = np.full_like(signal, np.nan)

        trimmed, start, end = trim_nan_edges(signal)
        valid = ~np.isnan(trimmed)

        # Interpolate NaNs before filtering
        if np.isnan(trimmed).any():
            trimmed = pd.Series(trimmed).interpolate(method='linear', limit_direction='both').values

        if len(trimmed) > 3 * max(len(a), len(b)):
            filtered = filtfilt(b, a, trimmed)
            filtered[~valid] = np.nan
            profile_filtered[start:end] = filtered

        binned_df[f'{var}_HP'] = profile_filtered
        binned_df['PROFILE_NUMBER'] = profile.PROFILE_NUMBER.values[0]
        return xr.Dataset.from_dataframe(binned_df.set_index('DEPTH'))

    # Process each profile
    profile_numbers = np.unique(ds.PROFILE_NUMBER.values)
    filtered = [
        filter_profile(ds.where(ds.PROFILE_NUMBER == pn, drop=True))
        for pn in tqdm(profile_numbers, desc=f"Filtering {var} by DEPTH")
    ]

    filtered = [f for f in filtered if f is not None]
    if not filtered:
        return xr.Dataset()
    result = xr.concat(filtered, dim="DEPTH")
    result = result.sortby('DEPTH')
    result.attrs = ds.attrs
    result[f'{var}_HP'].attrs = ds[var].attrs
    result[f'{var}_HP'].attrs['filter'] = 'highpass_depth'

    return result


def add_velocity_scale(ds, var='W_W_HP', window_size_seconds=100, axis="TIME"):
    """
    Compute RMS of velocity in a moving window for each profile and add it to the dataset as 'VELOCITY_SCALE'.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with velocity and TIME or DEPTH coordinate.
    var : str
        Name of the velocity variable to process.
    window_size_seconds : float
        Size of the moving window in seconds (if axis='TIME') or meters (if axis='DEPTH').
    axis : str, optional
        Axis along which to compute RMS ('TIME' or 'DEPTH').

    Returns
    -------
    ds : xr.Dataset
        The dataset with added 'SIGMA_W' and 'SIGMA_W2' variables.
    """
    profile_numbers = np.unique(ds.PROFILE_NUMBER.values)
    all_sigma_w = []
    dims = axis

    for profile_number in tqdm(profile_numbers, desc=f"Processing profiles ({axis})"):
        ds_profile = ds.where(ds.PROFILE_NUMBER == profile_number, drop=True)

        coord_vals = ds_profile[axis].values
        if len(coord_vals) < 2:
            all_sigma_w.append(xr.full_like(ds_profile[var], np.nan))
            continue

        # Compute sampling interval
        if axis == "TIME":
            deltas = np.diff(coord_vals) / np.timedelta64(1, 's')  # seconds
        elif axis == "DEPTH":
            deltas = np.diff(coord_vals)  # meters
        else:
            raise ValueError("axis must be 'TIME' or 'DEPTH'")

        deltas = deltas[np.isfinite(deltas)]
        deltas = deltas[deltas < (100 if axis == "TIME" else 10)]  # ignore large gaps
        mean_delta = np.nanmean(deltas) if len(deltas) else 1.0

        # Convert physical window size (s or m) to number of samples
        window_size = max(1, int(window_size_seconds / mean_delta))

        # Compute RMS of the high-passed velocity
        def rolling_rms(arr, window_size, dim):
            rolled = arr.rolling({dim: window_size}, center=True)
            constructed = rolled.construct(f"{dim}_window")
            return constructed.mean(f"{dim}_window", skipna=True) ** 0.5

        w_hp = ds_profile[var] / 100.0  # cm/s → m/s
        w_hp2 = w_hp ** 2
        sigma_w = rolling_rms(w_hp2, window_size, dim=axis)

        all_sigma_w.append(sigma_w)

    # Concatenate across profiles
    full_sigma_w = xr.concat(all_sigma_w, dim=axis)

    # Add to dataset
    ds['SIGMA_W'] = full_sigma_w
    ds['SIGMA_W'].attrs = {
        'long_name': 'Velocity scale (RMS)',
        'description': f'RMS of {var} in ±{window_size_seconds / 2} {axis.lower()} window per profile',
        'units': 'm/s',
        f'window_size_{axis.lower()}': window_size_seconds
    }
    ds['SIGMA_W2'] = ds['SIGMA_W'] ** 2
    ds['SIGMA_W2'].attrs = {
        'units': 'm^2/s^2',
        f'window_size_{axis.lower()}': window_size_seconds
    }

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

def add_adiabatic_sorted_N2(ds, plev = 20):
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
    vars = ["PRES","TEMP","PSAL","LATITUDE","LONGITUDE", "DEPTH"]
    if "TIME" in ds:
        vars.append("TIME")
    groups = utilities.group_by_profiles(ds, vars)
    print(f"Calculating adiabatic N² for {len(groups)} profiles...")
    df = groups.apply(sorted_N2_profile,plev=plev)

    dims = list(ds.sizes.keys())[0]
    
    SORTED_N_da = xr.DataArray(df['SORTED_N2'].to_numpy()**0.5, dims=dims, attrs={
        'long_name': 'Adiabatically sorted N',
        'units': '1/s',
        'plev': plev})

    ds['SORTED_N'] = SORTED_N_da

    return ds

def add_unsorted_N2(ds, var_rho='SIGTHETA'):
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
            if dz != 0:
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

def LEM_dissipation(ds, c=0.37):
    """
    Calculate turbulent kinetic energy dissipation rate using the Large Eddy Method (LEM):
        e = c * N * (q')^2

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
    # Compute dissipation
    velocity_scale = ds['SIGMA_W2']
    N = ds['SORTED_N']
    #N = ds['N']
    dissipation = c * N * velocity_scale
    
    ###mask all values below 1e-10
    #dissipation = dissipation.where(dissipation > 1e-10, drop=True)

    dissipation_log = np.log10(dissipation)

    # Add to dataset with metadata
    ds['E_GL'] = dissipation
    ds['E_GL'].attrs = {
        'long_name': 'Turbulent kinetic energy dissipation rate (LEM)',
        'description': r"Calculated as ε = c * N * $\sigma_w$² using Large Eddy Method",
        'units': 'W/kg',  # Assuming N in 1/s and q' in m/s
        'c_epsilon': c
    }
    ds.attrs['c_epsilon'] = c  # Store model constant in dataset attributes

    return ds

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
        mld_ds[var + '_MEAN'] = ('TIME', mean_values)

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

