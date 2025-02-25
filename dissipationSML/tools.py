import numpy as np
import gsw
import xarray as xr
import tqdm
import regionmask as rm

def add_pot_density_from_raw_data(ds: xr.Dataset):
    """
    This function computes the potential density and its anomaly with respect to the reference pressure of 0 dbar
    using the raw temperature and salinity data. The two new variables are added to the dataset.

    Parameters
    ----------
    ds: xarray dataset containing the raw temperature and salinity data

    Returns
    -------
    ds: xarray dataset with the additional variables SIG_THETA_RAW and SIGMA_T_RAW

    """
    PSAL = ds.PSAL_RAW.values
    TEMP = ds.TEMP_RAW.values
    PRES = ds.PRES.values
    lon = ds.LONGITUDE
    lat = ds.LATITUDE
    CT = gsw.CT_from_t(PSAL, TEMP, PRES)  # Conservative temperature
    SA = gsw.SA_from_SP(PSAL,TEMP, lon, lat)  # Absolute salinity  
    SIGTHETA_RAW = gsw.pot_rho_t_exact(SA, TEMP, PRES, 0)  # Potential density
    SIGMA_T_RAW = gsw.density.sigma0(SA, CT)

    ds['SIGTHETA_RAW'] = xr.DataArray(SIGTHETA_RAW, dims=('N_MEASUREMENTS'),
                         attrs={'units': 'kg/m^3', 'long_name': 'potential density with respect to 0 dbar'})
    ds['SIGMA_T_RAW'] = xr.DataArray(SIGMA_T_RAW, dims=('N_MEASUREMENTS'), 
                         attrs={'units': 'kg/m^3', 'long_name': 'potential density anomaly with respect to 0 dbar'})

    return ds


def bin_data(ds_profile: xr.Dataset, vars: list, resolution: float, agg: str = 'mean'):
    """
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
    """
    # Remove empty strings from vars list
    vars = [var for var in vars if var]

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
    else:
        MLD = linear_interpolation(density_below_10m, depth_below_10m, density_10m + threshold)
        return MLD

    # Find the mixed layer depth (only analyzing depths below 10m)
    #for i in range(len(density_below_10m)):
    #    if density_below_10m[i] > density_10m + threshold:
    #        return depth_below_10m[i]  # Return first depth that exceeds threshold
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
    # Compute the potential density anomaly σ₁(S,θ,z)
    sigma1 = density - sigma_0
    
    # Compute CR up to the reference depth h
    CR = calculate_CR_for_all_depth(depth, sigma1)
    
    # Find the depth where CR exceeds the critical value
    mask = np.array(CR) < - 0.1
    MLD = np.min(depth[mask])
    
    return MLD


def add_MLD_to_dataset(ds: xr.Dataset, use_raw: bool, use_bins: bool = False, binning: float = 1,agg: str = 'mean'):
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
    mld_array = np.full(len(ds.N_MEASUREMENTS), np.nan)
    # Get unique profile numbers in the dataset
    profile_numbers = np.unique(ds.PROFILE_NUMBER.values)

    # Initialize tqdm progress bar
    for profile_number in tqdm.tqdm(profile_numbers, desc="Calculating and adding MLD for Profiles", unit="profile"):
        profile = ds.where(ds.PROFILE_NUMBER == profile_number, drop=True)

        if profile.N_MEASUREMENTS.size == 0:
            continue  # Skip if no measurements exist for this profile
        if use_bins:
            depth, temperature, salinity, density = bin_data(profile, binning, use_raw, agg)
        else:
            if use_raw:
                density = profile.SIGMA_T_RAW.values
            else:
                density = profile.SIGMA_T.values
            depth = profile.DEPTH.values
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
    region_mask = region.mask(ds.LONGITUDE, ds.LATITUDE)
    ds_region = ds.isel(N_MEASUREMENTS=region_mask == 0)

    return ds_region