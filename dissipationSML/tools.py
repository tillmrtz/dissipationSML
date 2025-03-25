import numpy as np
import gsw
import xarray as xr
import tqdm
import regionmask as rm
from scipy.signal import convolve
from scipy.signal.windows import hann

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
    SA = gsw.SA_from_SP(PSAL,TEMP, lon, lat)  # Absolute salinity
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

"""
def bin_data(ds_profile: xr.Dataset, vars: list, resolution: float, agg: str = 'mean'):
    
    Bin the data in a profile dataset by depth using fixed depth steps.
    
    Parameters
    ----------
        ds_profile: xr.Dataset 
            The dataset containing at least **DEPTH and the variables to bin**.
        resolution: float 
            The depth resolution for the binning.
        var: list
            The variables to bin.
        agg: str 
            The aggregation method to use for binning. Default is 'mean'. Other option is 'median'.

    Returns
    -------
        dict: A dictionary containing binned data arrays for each variable, including DEPTH.

    Notes
    -----
    Original author: Till Moritz

    # Remove empty strings from vars list
    #vars = [var for var in vars if var]

    # Define bin edges and bin centers
    min_depth = np.floor(ds_profile.DEPTH.min() / resolution) * resolution
    max_depth = np.ceil(ds_profile.DEPTH.max() / resolution) * resolution
    bins = np.arange(min_depth, max_depth + resolution, resolution)
    bin_centers = bins[:-1] + resolution / 2  # Set depth values to bin centers
    
    binned_data = {}
    # Assign bin centers as the new depth values
    binned_data['DEPTH'] = bin_centers
    # Group variables by depth bins and apply aggregation
    for name in vars:
        if bins.size < 2:
            binned_data[name] = np.full(len(bin_centers), np.nan)
            continue
        grouped = ds_profile[name].groupby_bins('DEPTH', bins)
        if agg == 'mean':
            binned_data[name] = grouped.mean().values
        elif agg == 'median':
            binned_data[name] = grouped.median().values
        else:
            raise ValueError(f"Invalid aggregation method: {agg}")

    return binned_data
"""

def bin_data(ds_profile: xr.Dataset, vars: list, resolution: float, agg: str = 'mean'):
    """
    Bin the data in a profile dataset by depth using fixed depth steps, starting from 5m.

    Parameters
    ----------
    ds_profile : xr.Dataset
        The dataset containing at least **DEPTH and the variables to bin**.
    vars : list
        The variables to bin.
    resolution : float
        The depth resolution for binning.
    agg : str, optional
        The aggregation method ('mean' or 'median'). Default is 'mean'.

    Returns
    -------
    dict
        A dictionary containing binned data arrays for each variable, including DEPTH.

    Notes
    -----
    - Only depths greater than 5m are considered.
    - Bins start at 5m and follow the given resolution (e.g., 5-15m, 15-25m, ...).
    """

    # Consider only depths greater than 5m
    ds_filtered = ds_profile.where(ds_profile.DEPTH > 5, drop=True)

    if ds_filtered.DEPTH.size == 0:
        # If no valid depths remain, return NaNs
        return {var: np.full(1, np.nan) for var in vars + ['DEPTH']}

    # Define bin edges starting at 5m
    min_depth = 5
    max_depth = np.ceil(ds_filtered.DEPTH.max() / resolution) * resolution
    bins = np.arange(min_depth, max_depth + resolution, resolution)
    bin_centers = bins[:-1] + resolution / 2  # Center of each bin

    # Dictionary to store binned results
    binned_data = {'DEPTH': bin_centers}

    # Group variables by depth bins and apply aggregation
    for name in vars:
        grouped = ds_filtered[name].groupby_bins('DEPTH', bins)
        if agg == 'mean':
            binned_data[name] = grouped.mean().values
        elif agg == 'median':
            binned_data[name] = grouped.median().values
        else:
            raise ValueError(f"Invalid aggregation method: {agg}")

    return binned_data


def linear_interpolation(x, y, x_new):
    """
    Linearly interpolates the given x and y values to new x values.
    
    Parameters:
        x (numpy array): The x values to interpolate from.
        y (numpy array): The y values to interpolate from.
        x_new (numpy array): The new x values to interpolate to.
    
    Returns:
        numpy array: The interpolated y values at the new x values.
    """
    return np.interp(x_new, x, y)


def calculate_mixed_layer_depth(density: np.array, depth: np.array):
    """
    Computes the mixed layer depth (MLD) based on the density profile.
    The MLD is defined as the depth at which density exceeds the reference 
    density at 10m depth by 0.03 kg/m³ (Theory by Montégut et al., 2004 and used by Beaird et al., 2016 and Evans et al., 2018).
    
    Parameters
    ----------
    density : numpy array
        Density profile of the water column.
    depth : numpy array
        Corresponding depth values of the density profile.

    Returns
    -------
    mld : float
        Mixed layer depth of the profile (or NaN if not found).
    """
    # Remove NaN values from both arrays
    valid_mask = ~np.isnan(depth) & ~np.isnan(density)
    depth = depth[valid_mask]
    density = density[valid_mask]

    # Check for empty arrays after removing NaNs
    if depth.size == 0 or density.size == 0:
        return np.nan

    # Sort depth and density together
    sort_idx = np.argsort(depth)
    depth = depth[sort_idx]
    density = density[sort_idx]

    ### check if there exist the depth of 10m in the profile
    if 10 in depth:
        # Find the index of the depth closest to 10m
        depth_idx = np.nanargmin(abs(depth - 10))
        density_10m = density[depth_idx]
    else:
        # Interpolate the density at 10m depth with the two closest depth values
        density_10m = linear_interpolation(depth, density, 10)

    # Select only depths greater than 10m
    below_10m_mask = depth > 10
    depth_below_10m = depth[below_10m_mask]
    density_below_10m = density[below_10m_mask]

    # If no depths are below 10m, return NaN
    if depth_below_10m.size == 0:
        return np.nan

    # Density threshold
    threshold = 0.03

    ### find the first depth below 10m where the density exceeds the density at 10m by the threshold by using the linear interpolation
    if np.nanmax(density_below_10m) < density_10m + threshold:
        return np.nan
    #else:
    #    MLD = linear_interpolation(density_below_10m, depth_below_10m, density_10m + threshold)
    #    return MLD

    # Find the mixed layer depth (only analyzing depths below 10m)
    for i in range(len(density_below_10m)):
        if density_below_10m[i] > density_10m + threshold:
            return (depth_below_10m[i] + depth_below_10m[i - 1]) / 2
    #
    #return np.nan  # Return NaN if no depth satisfies the condition


def compute_CR(depth, h, sigma1):
    """
    Compute CR up to the reference depth h.
    
    Parameters:
    depth: array-like 
        Depth values corresponding to sigma1 in meters.
    h: float
        Reference depth up to which CR is computed.
    sigma1: array-like 
        Potential density anomaly σ₁(S,θ,z) in kg/m³.
    
    Returns:
    float: Computed CR up to the reference depth h.
    """

    # Ensure h is within the depth range
    if h > np.nanmax(depth):
        raise ValueError("h exceeds the available depth range!")
    
    ### Mask NaN values
    mask = ~np.isnan(depth) & ~np.isnan(sigma1)
    depth = depth[mask]
    sigma1 = sigma1[mask]

    # Sort depth and density together
    sort_idx = np.argsort(depth)
    depth = depth[sort_idx]
    sigma1 = sigma1[sort_idx]

    # Mask values where depth is shallower than h (i.e., between -h and 0)
    mask = (depth <= h) & (depth >= 0)

    # Integrate σ₁(S,θ,z) over depth using the trapezoidal rule
    integral = np.trapz(sigma1[mask], depth[mask])

    # Interpolate σ₁ at depth h
    sigma1_h = np.interp(h, depth, sigma1)

    # Compute CR(h)
    CR_h = integral - h * sigma1_h

    return CR_h

def calculate_CR_for_all_depth(depth,sigma1):
    CR = []
    for h in depth:
        CR_h = compute_CR(depth, h, sigma1)
        CR.append(CR_h)
    return CR

def calculate_MLD_with_CR(density: np.array, depth: np.array, sigma_0: float = 27.553558):
    """
    Calculate the mixed layer depth (MLD) using the convective resistance (CR) method.
    
    Parameters:
    density: array-like 
        Density values in kg/m³.
    depth: array-like 
        Depth values in meters.
    sigma_0: float
        Reference density in kg/m³.
    
    Returns:
    float: Mixed layer depth (MLD) in meters.
    """
    # Check for empty arrays after removing NaNs
    if depth.size == 0 or density.size == 0:
        return np.nan
    # Compute the potential density anomaly σ₁(S,θ,z)
    sigma1 = density - sigma_0
    
    # Compute CR up to the reference depth h
    CR = calculate_CR_for_all_depth(depth, sigma1)
    
    # Find the depth where CR exceeds the critical value
    mask = np.array(CR) < - 0.1
    MLD = np.min(depth[mask])
    
    return MLD

def add_MLD_to_dataset(ds: xr.Dataset, use_raw: bool = False, use_bins: bool = False, binning: float = 1, agg: str = 'mean'):
    """
    Computes the mixed layer depth for each profile in the dataset and adds it as a new variable.
    The value is stored equally for each measurement in each profile.
    If no mixed layer depth is found for a profile, the value is NaN.

    Parameters
    ----------
    ds: xarray dataset containing the potential density data

    Returns
    -------
    ds: xarray dataset with the additional variable MLD
    """
    # Create an empty array for MLD values
    mld_array = np.full(ds.dims['N_MEASUREMENTS'], np.nan)

    # Get unique profile numbers in the dataset
    profile_numbers = np.unique(ds.PROFILE_NUMBER.values)
    n_Measurements = 0
    for profile_number in tqdm.tqdm(profile_numbers, desc="Calculating and adding MLD for Profiles", unit="profile"):
        # Select measurements for the current profile
        profile = ds.where(ds.PROFILE_NUMBER == profile_number, drop=True)

        if profile.N_MEASUREMENTS.size == 0:
            continue  # Skip if no measurements exist for this profile

        if use_bins:
            binned_data = bin_data(profile, vars=['SIGMA_T'], resolution=binning, agg=agg)
            density, depth = binned_data['SIGMA_T'], binned_data['DEPTH']
        else:
            depth = profile.DEPTH.values
            density = profile.SIGMA_T_RAW.values if use_raw else profile.SIGMA_T.values

        # Compute MLD
        mld = calculate_mixed_layer_depth(density, depth)
        # Assign MLD to the correct indices
        for i in range(profile.N_MEASUREMENTS.size):
            mld_array[n_Measurements] = mld
            n_Measurements += 1

    # Add the computed MLD as a new variable in the dataset
    ds['MLD'] = xr.DataArray(mld_array, dims=('N_MEASUREMENTS'),
                             attrs={'units': 'm', 'long_name': 'Mixed layer depth',
                                    'data_used': 'Raw' if use_raw else 'Corrected',
                                    'binning': 'No binning' if not use_bins else f'{str(binning)}m bins',
                                    'aggregation': agg if use_bins else 'No binning'})

    return ds


"""
def add_MLD_to_dataset(ds: xr.Dataset, use_raw: bool = False, use_bins: bool = False, binning: float = 1,agg: str = 'mean'):
    
    Computes the mixed layer depth for each profile in the dataset and adds it as a new variable.
    The value is stored equally for each measurement in each profile.
    If no mixed layer depth is found for a profile, the value is NaN.

    Parameters
    ----------
    ds: xarray dataset containing the potential density data

    Returns
    -------
    ds: xarray dataset with the additional variable MLD
    

    mld_array = np.full(len(ds.N_MEASUREMENTS), np.nan)
    # Get unique profile numbers in the dataset
    profile_numbers = np.unique(ds.PROFILE_NUMBER.values)

    # Initialize tqdm progress bar
    for profile_number in tqdm.tqdm(profile_numbers, desc="Calculating and adding MLD for Profiles", unit="profile"):
        profile = ds.sel(N_MEASUREMENTS = ds.PROFILE_NUMBER == profile_number)

        if profile.N_MEASUREMENTS.size == 0:
            continue  # Skip if no measurements exist for this profile
        if use_bins:
            binned_data = bin_data(profile,vars=['SIGMA_T'],resolution=binning,agg=agg)
            density,depth = binned_data['SIGMA_T'],binned_data['DEPTH']
        else:
            depth = profile.DEPTH.values
            if use_raw:
                density = profile.SIGMA_T_RAW.values

            else:
                density = profile.SIGMA_T.values
        mld = calculate_mixed_layer_depth(density, depth)

        # Store MLD for the respective indices
        mld_array[profile.N_MEASUREMENTS.values] = mld

    # Add the computed MLD as a new variable
    ds['MLD'] = xr.DataArray(mld_array, dims=('N_MEASUREMENTS'),
                             attrs={'units': 'm', 'long_name': 'Mixed layer depth',
                                    'data_used': 'Raw' if use_raw else 'Corrected',
                                    'binning': 'No binning' if not use_bins else f'{str(binning)}m bins',
                                    'aggregation´': agg if use_bins else 'No binning'})
    return ds
"""

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

