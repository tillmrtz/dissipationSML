import numpy as np
import gsw
import xarray as xr

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


def bin_data(ds_profile: xr.Dataset, resolution, use_raw=False):
    """
    Bin depth, temperature, salinity, and compute density using the GSW package while preserving the input data shape.
    
    Parameters:
        ds_profile (xarray.Dataset): The dataset containing depth, temperature, and salinity data of one profile
        resolution (float): The depth resolution for binning.

    Returns:
        tuple: (binned_depths, binned_temperatures, binned_salinity, binned_density), where each is a numpy array with NaNs for unused indices.
    """
    if use_raw:
        temperature = ds_profile.TEMP_RAW.values
        salinity = ds_profile.PSAL_RAW.values
        density = ds_profile.SIGMA_T_RAW.values
    else:
        temperature = ds_profile.TEMP.values
        salinity = ds_profile.PSAL.values
        density = ds_profile.SIGMA_T.values

    depth = ds_profile.DEPTH.values
    pressure = ds_profile.PRES.values

    min_depth = np.floor(np.min(depth) / resolution) * resolution
    max_depth = np.ceil(np.max(depth) / resolution) * resolution
    bins = np.arange(min_depth, max_depth + resolution, resolution)
    
    unique_indices = np.zeros_like(depth, dtype=bool)
    
    binned_depths = np.full_like(depth, np.nan)
    binned_pressure = np.full_like(pressure, np.nan)

    binned_temperatures = np.full_like(temperature, np.nan)
    binned_salinity = np.full_like(salinity, np.nan)
    binned_density = np.full_like(density, np.nan)

    
    for i in range(len(bins) - 1):
        mask = (depth >= bins[i]) & (depth < bins[i + 1]) & ~unique_indices
        indices = np.where(mask)[0]
        
        if len(indices) > 0:
            unique_indices[indices] = True
            binned_depths[indices[0]] = (bins[i] + bins[i + 1]) / 2  # Bin center
            binned_temperatures[indices[0]] = np.mean(temperature[mask])
            binned_salinity[indices[0]] = np.mean(salinity[mask])
            binned_pressure[indices[0]] = np.mean(pressure[mask])
            binned_density[indices[0]] = np.mean(density[mask])

    return binned_depths, binned_temperatures, binned_salinity, binned_density

def calculate_mixed_layer_depth(density, depth):
    """
    Computes the mixed layer depth (MLD) based on the density profile.
    The MLD is defined as the depth at which density exceeds the reference 
    density at 10m depth by 0.03 kg/m³ (Evans et al., 2018).
    
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
    depth_idx = np.nanargmin(abs(depth - 10))  # Avoids issues with NaNs
    depth_10m = depth[depth_idx]
    density_10m = density[depth_idx]

    # Density threshold
    threshold = 0.03

    # Find the mixed layer depth
    for i in range(len(density)):
        if density[i] > density_10m + threshold:
            return depth[i]  # Return as soon as the threshold is exceeded

    return np.nan  # Return NaN if no depth satisfies the condition



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