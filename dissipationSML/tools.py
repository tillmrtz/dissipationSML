import numpy as np
import gsw
import xarray as xr
import tqdm

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


def bin_data(ds_profile: xr.Dataset, resolution: float, use_raw: bool =False, agg: str = 'mean'):
    """
    Bin depth, temperature, salinity, and compute density using the GSW package while preserving the input data shape.
    
    Parameters:
        ds_profile (xarray.Dataset): The dataset containing depth, temperature, and salinity data of one profile
        resolution (float): The depth resolution for binning.
        use_raw (bool): Whether to use raw temperature and salinity data.
        agg (str): The aggregation method to use for binning. Default is 'mean'. Other option is 'median'.

    Returns:
        tuple: (binned_depths, binned_temperatures, binned_salinity, binned_density), where each is a numpy array with NaNs for unused indices.
    """
    if use_raw:
        temperature = ds_profile.TEMP_RAW
        salinity = ds_profile.PSAL_RAW
        density = ds_profile.SIGMA_T_RAW
    else:
        temperature = ds_profile.TEMP
        salinity = ds_profile.PSAL
        density = ds_profile.SIGMA_T

    depth = ds_profile.DEPTH
    pressure = ds_profile.PRES
    ### group depth values into discrete intervals for analysis with the given resolution
    bins = np.arange(np.floor(np.min(depth) / resolution) * resolution,
                     np.ceil(np.max(depth) / resolution) * resolution + resolution,resolution)

    variables = {"pressure": pressure,"depths": depth,"temperatures": temperature,"salinity": salinity,"density": density}
    if agg == 'mean':
        binned_data = {name: var.groupby_bins('DEPTH', bins).mean().values for name, var in variables.items()}
    elif agg == 'median':
        binned_data = {name: var.groupby_bins('DEPTH', bins).median().values for name, var in variables.items()}
    else:
        raise ValueError(f"Invalid aggregation method: {agg}")
    
    return binned_data["depths"], binned_data["temperatures"], binned_data["salinity"], binned_data["density"]


def calculate_mixed_layer_depth(density, depth):
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

    # Find the index of the depth closest to 10m
    depth_idx = np.nanargmin(abs(depth - 10))
    density_10m = density[depth_idx]

    # Select only depths greater than 10m
    below_10m_mask = depth > 10
    depth_below_10m = depth[below_10m_mask]
    density_below_10m = density[below_10m_mask]

    # If no depths are below 10m, return NaN
    if depth_below_10m.size == 0:
        return np.nan

    # Density threshold
    threshold = 0.03

    # Find the mixed layer depth (only analyzing depths below 10m)
    for i in range(len(density_below_10m)):
        if density_below_10m[i] > density_10m + threshold:
            return depth_below_10m[i]  # Return first depth that exceeds threshold

    return np.nan  # Return NaN if no depth satisfies the condition


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
    max_profile = int(ds.PROFILE_NUMBER.max().values.item())

    # Initialize tqdm progress bar
    for profile_number in tqdm.tqdm(range(1, max_profile + 1), desc="Calculating and adding MLD for Profiles", unit="profile"):
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

    Original author
    ----------------
    Till Moritz
    """
    max_depths = ds.groupby('PROFILE_NUMBER').apply(lambda x: x['DEPTH'].max())
    min_depths = ds.groupby('PROFILE_NUMBER').apply(lambda x: x['DEPTH'].min())
    ### add the unit to the dataarray
    max_depths.attrs['units'] = ds['DEPTH'].attrs['units']
    min_depths.attrs['units'] = ds['DEPTH'].attrs['units']
    return min_depths, max_depths