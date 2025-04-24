import numpy as np
import gsw
import xarray as xr
import tqdm
import regionmask as rm
from scipy.signal import convolve
from scipy.signal.windows import hann
from scipy.integrate import cumulative_trapezoid
import pandas as pd

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


def bin_profile(ds_profile, vars, binning, agg: str = 'mean'):
    """
    Bins the data for a single profile using the construct_2dgrid function. The binning determines the depth resolution.

    Parameters
    ----------
    ds_profile : xr.Dataset or pd.DataFrame
        The dataset or dataframe containing the data of one profile containing at least 'DEPTH', 'PROFILE_NUMBER' and the variables to bin.
    vars : list
        The variables to bin.
    binning : float
        The depth resolution for binning.
    agg : str, optional
        The aggregation method ('mean' or 'median'). Default is 'mean'.

    Returns
    -------
    binned_profile: pd.DataFrame
        A dataframe containing the binned data for the selected profile.

    Notes
    -----
    Original author: Till Moritz
    """
    ## check if only one profile is selected
    if isinstance(ds_profile, xr.Dataset):
        profile_number = ds_profile['PROFILE_NUMBER'].values
    elif isinstance(ds_profile, pd.DataFrame):
        profile_number = ds_profile.index.values
    if len(np.unique(profile_number)) > 1:
        raise ValueError("Only one profile can be selected for binning.")
    
    binned_data = {}
    msk = ds_profile['DEPTH'].values > 0
    depth = ds_profile['DEPTH'].values[msk]
    profile_number = profile_number[msk]

    for var in vars:
        var_grid, prof_num_grid, depth_grid = construct_2dgrid(profile_number, depth, ds_profile[var].values[msk],
                                                                xi=1, yi=binning, x_bin_center=False, y_bin_center=True, agg=agg)
        binned_data[var] = var_grid[0]
    binned_data['DEPTH'] = depth_grid[0]
    binned_data['PROFILE_NUMBER'] = prof_num_grid[0]

    return pd.DataFrame(binned_data)

def bin_data(ds_profile, vars: list = ['TEMP','PSAL'], resolution: float = 10, agg: str = 'mean'):
    """
    Bin the data in a profile dataset or DataFrame by depth using fixed depth steps. The minimum depth is between
    0 and the binning resolution, and the maximum depth is between the maximum depth of the profile and the binning resolution.

    Parameters
    ----------
    ds_profile : xr.Dataset or pd.DataFrame
        The dataset or dataframe containing at least 'DEPTH' and the variables to bin.
    vars : list
        The variables to bin.
    resolution : float
        The depth resolution for binning.
    agg : str, optional
        The aggregation method ('mean' or 'median'). Default is 'mean'.

    Returns
    -------
    dict
        A dictionary containing binned data arrays for each variable, including 'DEPTH'.
    """

    # Remove empty strings from vars list
    vars = [var for var in vars if var]

    # Validate aggregation
    if agg not in ['mean', 'median']:
        raise ValueError(f"Invalid aggregation method: {agg}")

    # Handle xarray.Dataset
    if isinstance(ds_profile, xr.Dataset):

        # Define bin edges and bin centers
        min_depth = np.floor(ds_profile.DEPTH.min() / resolution) * resolution
        max_depth = np.ceil(ds_profile.DEPTH.max() / resolution) * resolution
        bins = np.arange(min_depth, max_depth + resolution, resolution)
        bin_centers = bins[:-1] + resolution / 2  # Set depth values to bin centers

        # Group variables by depth bins and apply aggregation
        binned_data = {}
        for name in vars:
            grouped = ds_profile[name].groupby_bins('DEPTH', bins)
            if agg == 'mean':
                binned_data[name] = grouped.mean().values
            elif agg == 'median':
                binned_data[name] = grouped.median().values
            else:
                raise ValueError(f"Invalid aggregation method: {agg}")

        # Assign bin centers as the new depth values
        binned_data['DEPTH'] = bin_centers

    # Handle pandas.DataFrame
    elif isinstance(ds_profile, pd.DataFrame):
        #df = ds_profile[ds_profile['DEPTH'] > 5].copy()
        df = ds_profile.copy()

        if df.empty:
            return {var: np.full(1, np.nan) for var in vars + ['DEPTH']}

        min_depth = np.floor(df['DEPTH'].min() / resolution) * resolution
        max_depth = np.ceil(df['DEPTH'].max() / resolution) * resolution
        bins = np.arange(min_depth, max_depth + resolution, resolution)
        bin_labels = bins[:-1] + resolution / 2
        df['DEPTH_BIN'] = pd.cut(df['DEPTH'], bins, labels=bin_labels)

        binned_data = {'DEPTH': bin_labels}

        for name in vars:
            if agg == 'mean':
                grouped = df.groupby('DEPTH_BIN',observed=False)[name].mean()
            else:
                grouped = df.groupby('DEPTH_BIN',observed=False)[name].median()

            # Align with bin labels
            binned_data[name] = grouped.reindex(bin_labels).to_numpy()

    else:
        raise TypeError("Input must be an xarray.Dataset or pandas.DataFrame")

    return binned_data

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
                 use_bins: bool = False, binning: float = 10):
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
        groups = group_by_profiles(ds, [variable, "DEPTH","TIME"])
        mld = groups.apply(mld_profile_treshhold, variable=variable, threshold=threshold,
                            ref_depth=ref_depth, use_bins=use_bins, binning=binning)
    elif method == 'CR':
        if variable != 'SIGMA_1':
            print(f"Warning: {variable} can not be used for convective resistance calulation. Instead use SIGMA_1 for CR calculation.")
            variable = 'SIGMA_1'
        groups = group_by_profiles(ds, [variable, "DEPTH","TIME"])
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

    # Remove NaNs
    valid = ~np.isnan(depth) & ~np.isnan(density)
    depth, density = depth[valid], density[valid]

    if depth.size == 0 or density.size == 0:
        print("No valid data available for MLD calculation.")
        return np.nan
    
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
            return (depth_below[i] + depth_below[i - 1]) / 2

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

    return np.nanmin(depth[below_threshold])


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


def match_era5_to_glider(ds_glider, ds_ERA5, lon_range=None, lat_range=None):
    """
    Matches ERA5 data to glider profiles, either using nearest points or averaging within a spatial range.

    Parameters
    ----------
    ds_glider : xarray.Dataset
        The dataset containing glider data.
    ds_ERA5 : xarray.Dataset
        The ERA5 dataset to be matched.
    lon_range : float or None, optional
        The range (in degrees) for longitude to average ERA5 data. If None, selects the nearest point.
    lat_range : float or None, optional
        The range (in degrees) for latitude to average ERA5 data. If None, selects the nearest point.

    Returns
    -------
    xarray.Dataset
        The matched ERA5 data with the same PROFILE_NUMBER dimension as the glider dataset.
    """
    if 'PROFILE_NUMBER' in ds_glider.dims:
        mean_lat = ds_glider.LATITUDE.values
        mean_lon = ds_glider.LONGITUDE.values
        mean_time = ds_glider.TIME.values
        profiles = ds_glider.PROFILE_NUMBER.values
    else:
        # Compute mean time, longitude, and latitude per profile
        mean_time = ds_glider.TIME.groupby(ds_glider.PROFILE_NUMBER).mean()
        mean_lon = ds_glider.LONGITUDE.groupby(ds_glider.PROFILE_NUMBER).mean()
        mean_lat = ds_glider.LATITUDE.groupby(ds_glider.PROFILE_NUMBER).mean()
        profiles = np.unique(ds_glider.PROFILE_NUMBER)

    if lon_range or lat_range:
        ds_matched_all = []
        for profile in tqdm(profiles, desc="Matching profiles"):
            lon = mean_lon.sel(PROFILE_NUMBER=profile, drop=True).values
            lat = mean_lat.sel(PROFILE_NUMBER=profile, drop=True).values
            time = mean_time.sel(PROFILE_NUMBER=profile, drop=True).values

            # Select nearest valid_time
            ds_matched = ds_ERA5.sel(valid_time=time, method="nearest")

            # Select longitude and latitude range if provided
            if lon_range:
                ds_matched = ds_matched.sel(longitude=slice(lon - lon_range, lon + lon_range))
            if lat_range:
                ds_matched = ds_matched.sel(latitude=slice(lat - lat_range, lat + lat_range))

            # Compute mean over the selected region
            ds_matched = ds_matched.mean(dim=["latitude", "longitude", "time"], skipna=True)

            # Add profile number, longitude, and latitude as coordinates
            ds_matched = ds_matched.assign_coords(PROFILE_NUMBER=profile, longitude=lon, latitude=lat)
            ds_matched_all.append(ds_matched)

        # Concatenate all profiles along PROFILE_NUMBER
        ds_matched = xr.concat(ds_matched_all, dim="PROFILE_NUMBER")
    else:
        # Match without tolerance (nearest selection)
        ds_matched = ds_ERA5.sel(valid_time=mean_time, longitude=mean_lon, latitude=mean_lat, method="nearest")
        ds_matched = ds_matched.mean(dim="time")

    # Preserve attributes
    ds_matched.attrs = ds_ERA5.attrs
    ds_matched.attrs["longitude_range_used"] = f"±{lon_range} degrees" if lon_range else "Nearest longitude point"
    ds_matched.attrs["latitude_range_used"] = f"±{lat_range} degrees" if lat_range else "Nearest latitude point"

    for var in ds_ERA5.variables:
        ds_matched[var].attrs = ds_ERA5[var].attrs

    return ds_matched

def hann_window_filter(ds, vars: list, window_size):
    """
    Applies a Hann window filter (smoothing) to a variable in an xarray Dataset 
    and returns the dataset with the filtered variable.

    Parameters
    ----------
    ds : xarray.Dataset
        The input dataset containing the variable.
    vars : list
        A list of variable names to filter.
    window_size : int
        The size of the Hann window (must be an odd number).

    Returns
    -------
    xarray.Dataset
        The dataset with the filtered variable added.
    """
    if window_size % 2 == 0:
        raise ValueError("Window size must be an odd number.")

    # Generate the Hann window
    window = hann(window_size, sym=True)
    window /= window.sum()  # Normalize the window

    ds_filtered = ds.copy()
    # Apply convolution along the first axis for each variable
    for var in vars:
        # Apply the convolution along the 'valid_time' axis
        filtered_data = convolve(ds[var], window, mode="same", method="auto")

        # Store the filtered result as a new variable
        ds_filtered[f"{var}_hann_filtered"] = xr.DataArray(
            filtered_data, dims=ds[var].dims, coords=ds[var].coords, name=f"{var}_hann_filtered"
        )

    return ds_filtered

