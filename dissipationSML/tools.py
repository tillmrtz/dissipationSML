import numpy as np
import gsw
import xarray as xr

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
    else:
        temperature = ds_profile.TEMP.values
        salinity = ds_profile.PSAL.values

    depth = ds_profile.DEPTH.values
    pressure = ds_profile.PRES.values
    lon = ds_profile.LONGITUDE
    lat = ds_profile.LATITUDE

    min_depth = np.floor(np.min(depth) / resolution) * resolution
    max_depth = np.ceil(np.max(depth) / resolution) * resolution
    bins = np.arange(min_depth, max_depth + resolution, resolution)
    
    unique_indices = np.zeros_like(depth, dtype=bool)
    
    binned_depths = np.full_like(depth, np.nan)
    binned_temperatures = np.full_like(temperature, np.nan)
    binned_salinity = np.full_like(salinity, np.nan)
    binned_pressure = np.full_like(pressure, np.nan)
    
    for i in range(len(bins) - 1):
        mask = (depth >= bins[i]) & (depth < bins[i + 1]) & ~unique_indices
        indices = np.where(mask)[0]
        
        if len(indices) > 0:
            unique_indices[indices] = True
            binned_depths[indices[0]] = (bins[i] + bins[i + 1]) / 2  # Bin center
            binned_temperatures[indices[0]] = np.mean(temperature[mask])
            binned_salinity[indices[0]] = np.mean(salinity[mask])
            binned_pressure[indices[0]] = np.mean(pressure[mask])

    CT = gsw.CT_from_t(binned_salinity, binned_temperatures, binned_pressure)  # Conservative temperature
    SA = gsw.SA_from_SP(binned_salinity, binned_pressure, lon, lat)  # Absolute salinity
    binned_density = gsw.rho(SA, CT, binned_pressure)  # Density
    
    return binned_depths, binned_temperatures, binned_salinity, binned_density


def calculate_mixed_layer_depth(density_profile, depth_profile):
    reference_density = density_profile[0]  # Density at 10 m depth
    threshold_density = reference_density + 0.03  # Density threshold
    
    for i in range(len(density_profile)):
        if density_profile[i] > threshold_density:
            return depth_profile[i]  # Mixed layer depth reached
    
    return None  # Mixed layer depth not reached