import numpy as np
import gsw
import xarray as xr
from tqdm import tqdm
import regionmask as rm
from scipy.signal import convolve
from scipy.signal.windows import hann
from scipy.integrate import cumulative_trapezoid
import pandas as pd
from scipy.signal import butter, filtfilt

def add_pot_densities(ds: xr.Dataset, use_raw: bool = True):
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
    if use_raw:
        vars = [var + '_RAW' for var in vars]
    
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
        if use_raw:
            var_name = var + '_RAW'
        else:
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

    #ds['SIGTHETA_RAW'] = xr.DataArray(SIGTHETA_RAW, dims=('N_MEASUREMENTS'),
    #                     attrs={'units': 'kg/m^3', 'long_name': 'potential density with respect to 0 dbar'})
    #ds['SIGMA_T_RAW'] = xr.DataArray(SIGMA_T_RAW, dims=('N_MEASUREMENTS'), 
    #                     attrs={'units': 'kg/m^3', 'long_name': 'potential density anomaly with respect to 0 dbar'})
    #ds['SIGMA_1_RAW'] = xr.DataArray(SIGMA_1_RAW, dims=('N_MEASUREMENTS'),
    #                        attrs={'units': 'kg/m^3', 'long_name': 'potential density anomaly with respect to 1000 dbar'})

    return ds

def add_vertical_water_velocity(ds: xr.Dataset) -> xr.Dataset:
    """
    Compute and add total vertical water velocity (W) to the dataset.

    W is estimated from the pressure rate of change and the glider vertical velocity model.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing 'DEPTH', 'TIME', and 'GLIDER_VERT_VELO_MODEL'.

    Returns
    -------
    ds : xarray.Dataset
        Dataset with added 'VERTICAL_WATER_VELOCITY' and 'VERTICAL_VELOCITY_MEASURED' variables.
    """
    # Extract variables
    time = ds['TIME'].values
    depth = ds['DEPTH'].values
    glider_velo = ds['GLIDER_VERT_VELO_MODEL'].values
    # Clean glider velocity data
    #glider_velo[np.abs(glider_velo) > 20] = np.nan

    # Calculate vertical velocity from depth change (central difference, cm/s)
    ddepth = -(depth[2:] - depth[:-2]) * 100  # cm
    dtime = (time[2:] - time[:-2]) / np.timedelta64(1, 's')  # s

    # Handle invalid time intervals
    dtime[(dtime == 0) | (dtime > 500)] = np.nan

    # Estimate measured vertical velocity
    w_meas = ddepth / dtime
    w_meas = np.concatenate(([np.nan], w_meas, [np.nan]))  # Pad ends with NaN
    #w_meas[np.abs(w_meas) > 25] = np.nan  # Filter unrealistic values


    # Estimate vertical water velocity
    w_water = w_meas - glider_velo
    #w_water[np.abs(w_water) > 5] = np.nan
    w_water[depth < 10] = np.nan  # Ignore shallow data

    # Create DataArrays with metadata
    da_w_meas = xr.DataArray(
        w_meas,
        dims=ds['GLIDER_VERT_VELO_MODEL'].dims,
        coords=ds['GLIDER_VERT_VELO_MODEL'].coords,
        name="VERTICAL_WATER_VELOCITY_MEASURED",
        attrs={
            "units": "cm/s",
            "description": "Measured vertical velocity from depth change"
        }
    )

    da_w_water = xr.DataArray(
        w_water,
        dims=ds['GLIDER_VERT_VELO_MODEL'].dims,
        coords=ds['GLIDER_VERT_VELO_MODEL'].coords,
        name="VERTICAL_WATER_VELOCITY",
        attrs={
            "units": "cm/s",
            "description": "Vertical water velocity (measured - glider model)"
        }
    )

    # Add results to dataset
    ds['VERTICAL_VELOCITY_MEASURED'] = da_w_meas
    ds['VERTICAL_WATER_VELOCITY'] = da_w_water

    return ds

def rms_in_mld(ds: xr.Dataset, mld_ds: xr.Dataset, vars: list, min_depth: float) -> xr.Dataset:
    """
    Calculate the Root Mean Square (RMS) of variables between min_depth and the mixed layer depth (MLD) for each profile.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing variables (e.g., 'TEMP', 'PSAL', etc.) and 'DEPTH'.
    mld_ds : xarray.Dataset
        Dataset containing 'MLD' and 'PROFILE_NUMBER'.
    vars : list of str
        Variable names for which to compute the RMS within the MLD range.
    min_depth : float
        Minimum depth to start the RMS calculation.

    Returns
    -------
    xarray.Dataset
        The input MLD dataset with new variables added: '<var>_RMS' for each input variable.
    """
    mld_ds = mld_ds.copy()

    for var in vars:
        print(f"Calculating RMS for {var}...")
        rms_values = []

        for i in tqdm(range(len(mld_ds['PROFILE_NUMBER']))):
            profile_number = mld_ds['PROFILE_NUMBER'].values[i]
            mld_depth = mld_ds['MLD'].values[i]

            if np.isnan(mld_depth):
                rms_values.append(np.nan)
                continue

            # Extract variable and depth for current profile
            profile_mask = ds['PROFILE_NUMBER'] == profile_number
            profile_values = ds[var].where(profile_mask, drop=True)
            profile_depth = ds['DEPTH'].where(profile_mask, drop=True)

            # Mask to depths within min_depth and MLD
            valid_mask = (profile_depth >= min_depth) & (profile_depth <= mld_depth)
            values_in_range = profile_values.where(valid_mask, drop=True)

            # Compute RMS and store
            rms = np.sqrt((values_in_range ** 2).mean().values)
            rms_values.append(rms)

        # Add the result to the MLD dataset
        mld_ds[var + '_RMS'] = ('TIME', rms_values)

    return mld_ds

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

import numpy as np
import xarray as xr
from scipy.signal import butter, filtfilt
from tqdm import tqdm
import pandas as pd

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

        fs = 1 / mean_dt
        fc = 1 / cutoff_period
        wn = fc / (fs / 2)
        b, a = butter(order, wn, btype='high')

        binned_df = bin_profile(profile, [var, 'DEPTH'], binning=mean_dt, dim='TIME')
        signal = binned_df[var].values

        profile_filtered = np.full_like(signal, np.nan)
        trimmed, start, end = trim_nan_edges(signal)

        # Interpolate NaNs before filtering
        if np.isnan(trimmed).any():
            trimmed = pd.Series(trimmed).interpolate(
                method='linear', limit_direction='both').values

        if len(trimmed) > 3 * max(len(a), len(b)):
            filtered = filtfilt(b, a, trimmed)
            profile_filtered[start:end] = filtered

        binned_df[f'{var}_filtered'] = profile_filtered
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
    result[f'{var}_filtered'].attrs = ds[var].attrs
    result[f'{var}_filtered'].attrs['filter'] = 'highpass_time'

    return result

def highpass_butterworth(ds_binned, var, cutoff_wavelength=30, order=4):
    """
    Calculates a highpass Butterworth filter for a given variable in the dataset.
    
    Parameters
    ----------
    ds_binned: xr.Dataset
        Binned dataset containing the variable to be filtered.
    var: str
        Variable name to be filtered.
    cutoff_wavelength: float
        Cutoff wavelength in meters. Default is 30 m.
    order: int
        Order of the Butterworth filter. Default is 4.
    Returns
    -------
    ds_binned: xr.Dataset
        Binned dataset with the filtered variable added.
    """
    dz = ds_binned.attrs['binning']
    fs = 1 / dz
    fc = 1 / cutoff_wavelength
    fn = fs / 2
    normalized_cutoff = fc / fn
    b, a = butter(order, normalized_cutoff, btype='high', analog=False)

    profile_numbers = np.unique(ds_binned.PROFILE_NUMBER.values)

    full_filtered = []
    for profile_number in profile_numbers:
        profile = ds_binned.sel(TIME = ds_binned.PROFILE_NUMBER == profile_number)

        var_data = profile[var].values

        trimmed_var, start_idx, end_idx = trim_nan_edges(var_data)
        profile_filtered = np.full_like(var_data, np.nan)

        if len(trimmed_var) > 3 * max(len(a), len(b)):
            filtered_segment = filtfilt(b, a, trimmed_var)
            profile_filtered[start_idx:end_idx] = filtered_segment
        else:
            print(f"Skipping profile {profile_number}: too short for filtering")

        full_filtered.append(profile_filtered)
    full_filtered = np.concatenate(full_filtered)
    ds_binned[var + '_filtered'] = (('TIME'), full_filtered)
    ds_binned[var + '_filtered'].attrs = ds_binned[var].attrs
    ds_binned[var + '_filtered'].attrs['filter'] = 'highpass'

    return ds_binned


def construct_2dgrid(x, y, v, xi=1, yi=1, x_bin_center: bool = True, y_bin_center: bool = True, agg: str = 'median'):

    """
    Constructs a 2D gridded representation of input data based on specified resolutions. The function takes in x, y, and v data,
    and generates a grid where each cell contains the aggregated value (e.g., mean, median) of v corresponding to the x and y coordinates.
    If the input data is already binned and you want the grid coordinates to align with the original bin edges, set `x_bin_center` and `y_bin_center` to False and the 
    resolution (i.e. xi and yi) to the bin size.

    Parameters
    ----------
    x : array-like  
        Input data representing the x-dimension.  
    y : array-like  
        Input data representing the y-dimension.  
    v : array-like  
        Input data representing the z-dimension (values to be gridded).  
    xi : int or float, optional, default=1  
        Resolution for the x-dimension grid spacing.  
    yi : int or float, optional, default=1  
        Resolution for the y-dimension grid spacing.
    x_bin_center : bool, optional, default=True
        If True, the x-coordinate grid (`XI`) corresponds to the **center** of each x-bin.
        If False, it corresponds to the **left edge** of each bin.
        This is especially useful if the input `x` data is already binned with the same resolution as `xi`,
        and you want the grid coordinates to align with the original bin edges. (e.g. profile numbers).
    y_bin_center : bool, optional, default=True
        Same as `x_bin_center`, but for the y-coordinate grid (`YI`).
        Set to False if your `y` data is already pre-binned with the same resolution as `yi`.
    agg : str, optional, default='median'
        Aggregation method to be used for gridding. Options include 'mean', 'median', etc.

    Returns
    -------
    grid : numpy.ndarray  
        Gridded representation of the z-values over the x and y space.  
    XI : numpy.ndarray  
        Gridded x-coordinates corresponding to the specified resolution.  
    YI : numpy.ndarray  
        Gridded y-coordinates corresponding to the specified resolution. 

    Notes
    -----
    Original Author: Bastien Queste
    [Source Code](https://github.com/bastienqueste/gliderad2cp/blob/de0652f70f4768c228f83480fa7d1d71c00f9449/gliderad2cp/process_adcp.py#L140)
    
    Modified by Till Moritz: added the aggregation parameter and the option to chose either bin center or bin edge as the grid coordinates.
    """
    if np.size(xi) == 1:
        xi = np.arange(np.nanmin(x), np.nanmax(x) + xi+1, xi)
    if np.size(yi) == 1:
        yi = np.arange(np.nanmin(y), np.nanmax(y) + yi+1, yi)

    raw = pd.DataFrame({'x': x, 'y': y, 'v': v}).dropna()
    grid = np.full([len(xi)-1, len(yi)-1], np.nan)

    raw['xbins'], xbin_iter = pd.cut(raw.x, xi, retbins=True, labels=False, include_lowest=True, right=False)
    raw['ybins'], ybin_iter = pd.cut(raw.y, yi, retbins=True, labels=False, include_lowest=True, right=False)

    raw = raw.dropna(subset=['xbins', 'ybins'])  # Remove out-of-bound rows
    _tmp = raw.groupby(['xbins', 'ybins'])['v'].agg(agg)
    grid[_tmp.index.get_level_values(0).astype(int), _tmp.index.get_level_values(1).astype(int)] = _tmp.values
    # Match XI and YI shape to grid using bin centers
    if x_bin_center:
        xi = xi[:-1] + np.diff(xi) / 2
    else:
        xi = xi[:-1]
    if y_bin_center:
        yi = yi[:-1] + np.diff(yi) / 2
    else:
        yi = yi[:-1]
    YI, XI = np.meshgrid(yi, xi)
    return grid, XI, YI


def bin_profile(ds_profile, vars, binning, dim='DEPTH', agg='mean'):
    """
    Bin a single profile dataset along a specified dimension ('DEPTH' or 'TIME').

    Parameters
    ----------
    ds_profile : xr.Dataset or pd.DataFrame
        Profile data with at least 'DEPTH', 'TIME', and target variables.
    vars : list of str
        Variables to be binned.
    binning : float
        Bin size in meters (DEPTH) or seconds (TIME).
    dim : str, default 'DEPTH'
        Dimension along which to bin ('DEPTH' or 'TIME').
    agg : str, default 'mean'
        Aggregation method: 'mean' or 'median'.

    Returns
    -------
    binned_profile : pd.DataFrame
        Binned profile data as DataFrame.
    """

    # Ensure single profile
    profile_number = ds_profile['PROFILE_NUMBER'].values if isinstance(ds_profile, xr.Dataset) else ds_profile.index.values
    if len(np.unique(profile_number)) > 1:
        raise ValueError("Only one profile can be selected for binning.")

    binned_data = {}

    # Mask and extract dimension values
    if dim == 'DEPTH':
        mask = ds_profile[dim].values > 0
        dim_values = ds_profile[dim].values[mask]
    elif dim == 'TIME':
        dim_values = ds_profile[dim].values.astype('float64') * 1e-9  # convert ns to s
        mask = np.isfinite(dim_values)
    else:
        raise ValueError("`dim` must be either 'DEPTH' or 'TIME'")

    profile_number = profile_number[mask]

    # Handle very short or empty input
    if len(dim_values) <= 1 or any(len(ds_profile[var]) <= 1 for var in vars):
        return pd.DataFrame(columns=vars + [dim, 'PROFILE_NUMBER'])

    # Bin each variable
    for var in vars:
        var_grid, profile_grid, bin_grid = construct_2dgrid(
            profile_number, dim_values,ds_profile[var].values[mask],
            xi=1, yi=binning,
            x_bin_center=False, y_bin_center=True,
            agg=agg
        )
        binned_data[var] = var_grid[0]

    binned_data[dim] = bin_grid[0] * 1e9 if dim == 'TIME' else bin_grid[0]  # convert back to ns if needed
    binned_data['PROFILE_NUMBER'] = profile_grid[0]

    df = pd.DataFrame(binned_data)
    if 'TIME' in df.columns or 'TIME' in df.index:
        df['TIME'] = pd.to_datetime(df['TIME'], unit='ns', errors='coerce')

    if dim == 'DEPTH':
        if 'TIME' in df.columns:
            df = df.set_index('DEPTH')
            df['TIME'] = df['TIME'].interpolate(method='linear', limit_direction='both')
            df = df.reset_index()
    
    return df

def bin_all_profiles(ds, vars, binning, agg='mean', dim='DEPTH'):
    """
    Bin all profiles in the dataset along a specified dimension.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset with vertical or time-series profiles.
    vars : list of str
        Variables to be included in the binning.
    binning : float
        Bin size (in meters for DEPTH or seconds for TIME).
    agg : str, default 'mean'
        Aggregation method: 'mean' or 'median'.
    dim : str, default 'DEPTH'
        Dimension along which to bin: 'DEPTH' or 'TIME'.

    Returns
    -------
    ds_binned : xr.Dataset
        Binned dataset.
    """

    if agg not in ['mean', 'median']:
        raise ValueError("agg must be 'mean' or 'median'")

    required_vars = ['DEPTH', 'TIME', 'LONGITUDE', 'LATITUDE']
    all_vars = list(set(vars + required_vars))

    # Group dataset by profiles
    grouped = group_by_profiles(ds, all_vars)

    # Apply binning to each profile
    binned_df = grouped.apply(bin_profile, vars=all_vars, binning=binning, agg=agg, dim=dim)

    # Clean up DataFrame
    if 'PROFILE_NUMBER' in binned_df.columns:
        binned_df = binned_df.drop('PROFILE_NUMBER', axis=1)
        binned_df = binned_df.reset_index()
    if 'level_1' in binned_df.columns:
        binned_df = binned_df.drop('level_1', axis=1)
    if 'TIME' in binned_df.columns:
        binned_df = binned_df.set_index('TIME')

    # Convert to xarray Dataset
    ds_binned = xr.Dataset.from_dataframe(binned_df)

    # Convert time to datetime format
    if 'TIME' in ds_binned:
        ds_binned['TIME'] = ds_binned['TIME'].astype('datetime64[ns]')

    # Copy over variable and global attributes
    for var in ds_binned.data_vars:
        ds_binned[var].attrs = ds[var].attrs if var in ds else {}
    ds_binned.attrs = ds.attrs.copy()
    ds_binned.attrs['binning'] = binning
    ds_binned.attrs['binning_method'] = agg

    ds_binned = ds_binned.sortby('TIME')

    return ds_binned


def group_by_profiles(ds, variables=None):
    """
    Group glider dataset by the dive profile number.

    This function groups the dataset by the `PROFILE_NUMBER` column, where each group corresponds to a single profile.
    The resulting groups can be evaluated statistically, for example using methods like `pandas.DataFrame.mean` or
    other aggregation functions. To filter a specific profile, use `xarray.Dataset.where` instead.

    Parameters
    ----------
    ds : xarray.Dataset
        A 1-dimensional glider dataset containing profile information.
    variables : list of str, optional
        A list of variable names to group by, if only a subset of the dataset should be included in the grouping.
        Grouping by a subset is more memory-efficient and faster.

    Returns
    -------
    profiles : pandas.core.groupby.DataFrameGroupBy
        A pandas `GroupBy` object, grouped by the `PROFILE_NUMBER` of the glider dataset.
        This can be further aggregated using methods like `mean` or `sum`.

    Notes
    -----
    This function is based on the original GliderTools implementation and was modified by
    Chiara Monforte to ensure compliance with the OG1 standards.
    [Source Code](https://github.com/GliderToolsCommunity/GliderTools/blob/master/glidertools/utils.py)
    """
    ds = ds.reset_coords().to_pandas().reset_index().set_index("PROFILE_NUMBER")
    if variables:
        return ds[variables].groupby("PROFILE_NUMBER")
    else:
        return ds.groupby("PROFILE_NUMBER")

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
        groups = group_by_profiles(ds, [variable, "DEPTH","TIME","LONGITUDE","LATITUDE"])
        mld = groups.apply(mld_profile_treshhold, variable=variable, threshold=threshold,
                            ref_depth=ref_depth, use_bins=use_bins, binning=binning)
    elif method == 'CR':
        if variable != 'SIGMA_1':
            print(f"Warning: {variable} can not be used for convective resistance calulation. Instead use SIGMA_1 for CR calculation.")
            variable = 'SIGMA_1'
        groups = group_by_profiles(ds, [variable, "DEPTH","TIME","LONGITUDE","LATITUDE"])
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
        profile = bin_profile(profile, [variable], binning=binning)
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
    
    if np.nanmean(np.diff(depth)) < 0:
        depth = -1 * depth

    # Sort by depth
    sort_idx = np.argsort(depth)
    depth, density = depth[sort_idx], density[sort_idx]

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
        profile = bin_profile(profile, ['SIGMA_1'], binning=binning)

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
    if "langitude" in ds.coords:
        region_mask = region.mask(ds.longitude,ds.latitude)

    else:
        region_mask = region.mask(ds.LONGITUDE, ds.LATITUDE)
        ds_region = ds.isel(N_MEASUREMENTS=region_mask == 0)

    return ds_region


def match_era5_to_mld(ds_mld, ds_era5, lon_range=None, lat_range=None):
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

    Returns
    -------
    xarray.Dataset
        MLD dataset with ERA5 variables added as 1D arrays aligned with TIME.
    """
    times = ds_mld.TIME.values
    lons = ds_mld.LONGITUDE.values
    lats = ds_mld.LATITUDE.values

    matched_profiles = []

    for i in tqdm(range(len(times)), desc="Matching ERA5 to MLD"):
        time = times[i]
        lon = lons[i]
        lat = lats[i]

        # Select nearest ERA5 time
        match = ds_era5.sel(valid_time=time, method="nearest")

        # Apply spatial window if specified
        if lon_range:
            match = match.sel(longitude=slice(lon - lon_range, lon + lon_range))
        else:
            match = match.sel(longitude=lon, method="nearest")

        if lat_range:
            match = match.sel(latitude=slice(lat - lat_range, lat + lat_range))
        else:
            match = match.sel(latitude=lat, method="nearest")

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
    Q_SW = ds['ssr']/3600 # net shortwave radiation in W/m2
    Q_LW = ds['str']/3600 # net longwave radiation in W/m2
    Q_LH = ds['slhf']/3600 # net latent heat flux in W/m2
    Q_SH = ds['sshf']/3600 # net sensible heat flux in W/m2
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

def dissipation_bouyancy_flux(ds):
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
    
    ds = add_buoyancy_flux(ds)
    
    # Calculate dissipation rate based on buoyancy flux for all positive values of B_0
    B_0 = ds['B_0']
    B_0 = B_0.where(B_0 > 0 , 0)  # Set negative values to zero
    MLD = ds['MLD']
    ds['epsilon_Q'] = 1/2 * MLD * B_0
    
    return ds

def drag_coefficient_trenberth1990(U10):
    """Calculate neutral drag coefficient from Trenberth et al. (1990) based on wind speed (U10 in m/s)."""
    C_D = np.where(
        U10 > 10,
        (0.49 + 0.065 * U10) / 1000,
        np.where(
            U10 >= 3,
            1.14 / 1000,
            (0.62 + 1.56 / U10) / 1000
        )
    )
    return C_D

def add_wind_stress(ds,rho_air=1.225, Cd=0.0013):
    """
    Add wind stress to the dataset based on the wind speed and friction velocity.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing wind speed and friction velocity.
    rho_air : float, optional
        Density of air in kg/m^3 (default is 1.225 kg/m^3).
    Cd : float, optional
        Set the drag coefficient (default is 0.0013). If set to None, it will be calculated based Trenberth et al. (1990).
    
    Returns
    -------
    xarray.Dataset
        Dataset with wind stress added.
    """
    
    # Calculate wind stress based on wind speed and friction velocity
    u10 = ds['u10']  # Eastward wind speed at 10m height in m/s
    v10 = ds['v10']  # Northward wind speed at 10m height in m/s

    U = np.sqrt(u10**2 + v10**2)  # Wind speed in m/s

    if Cd is None:
        Cd = drag_coefficient_trenberth1990(U) # Drag coefficient based on wind speed
    
    # Calculate wind stress using bulk formula
    TAU = rho_air * Cd * U**2  # Wind stress in N/m^2
    tau = ds['tau']  # Wind stress in N/m^2
    rho = ds['SIGTHETA_MEAN']
    # Calculate friction velocity
    u_star = np.sqrt(tau / rho)  # Friction velocity in m/s from surface stress (ERA-5)
    U_STAR = np.sqrt(TAU / rho)  # Friction velocity in m/s from bulk formula
    
    ds['u_star'] = u_star
    ds['U_STAR'] = U_STAR
    ds['TAU'] = TAU

    return ds

def add_hs(ds):
    """
    Add the transition depth (Stokes depth) hs to the dataset. The calculation is based on Buckingham 2019.
    """
    g = 9.81 # gravitational acceleration
    kappa = 0.4 # von Karman constant
    T = ds['pp1d'] # peak wave period
    c_p = g*T/(2*np.pi) # phase speed of peak wave
    c_bar = 0.1*c_p # effective wave speed

    H_s = ds['swh'] # significant wave height
    u_star = ds['u_star'] # friction velocity from wind stress
    U_STAR = ds['U_STAR'] # friction velocity from bulk formula

    # Calculate the transition depth
    h_s = 0.3 * kappa * H_s * c_bar/u_star
    H_S = 0.3 * kappa * H_s * c_bar/U_STAR
    ds['h_s'] = h_s
    ds['H_S'] = H_S
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
    
    ds = add_wind_stress(ds)
    ds = add_hs(ds)
    
    # Extract variables from dataset
    MLD = ds['MLD']             # Mixed layer depth [m]
    tau = ds['tau']             # Wind stress [N/m^2]
    rho = ds['SIGTHETA_MEAN']   # Density [kg/m^3]
    h_s = ds['h_s']             # Transition depth [m]
    H_S = ds['H_S']             # Stokes depth [m]

    # Physical constant
    kappa = 0.4  # Von Karman constant
    u_star = ds['u_star']  # Friction velocity [m/s]
    U_STAR = ds['U_STAR']  # Friction velocity from bulk formula [m/s]

    # Compute epsilon_tau using conditional logic
    # Where h_s > MLD, epsilon_tau = nan; else compute full expression
    epsilon_tau = xr.where(h_s > MLD , 0 , 
                           (u_star ** 3) / kappa * np.log(MLD / h_s))
    EPSILON_TAU = xr.where(H_S > MLD , 0 , 
                           (U_STAR ** 3) / kappa * np.log(MLD / H_S))
    ds['epsilon_tau'] = epsilon_tau
    ds['EPSILON_TAU'] = EPSILON_TAU
    
    return ds