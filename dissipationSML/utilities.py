import numpy as np
import pandas as pd
import xarray as xr
import cmocean.cm as cmo
import matplotlib.cm as cm


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


def bin_profile(ds_profile, vars, binning = None, dim='DEPTH', agg='mean', max_interval = 10):
    """
    Bin a single profile dataset along a specified dimension ('DEPTH' or 'TIME').

    Parameters
    ----------
    ds_profile : xr.Dataset or pd.DataFrame
        Profile data with at least 'DEPTH', 'TIME', and target variables.
    vars : list of str
        Variables to be binned.
    binning : float
        Bin size in meters (DEPTH) or seconds (TIME). If binning is None, then the mean of the dimension is taken in each profile.
    dim : str, default 'DEPTH'
        Dimension along which to bin ('DEPTH' or 'TIME').
    agg : str, default 'mean'
        Aggregation method: 'mean' or 'median'.
    max_interval: float
        


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

    if binning == None:
        ddim = np.diff(dim_values) 
        ddim[ddim > max_interval] = np.nan
        if np.all(np.isnan(ddim)):
            print(f"All binning intervals of {dim} are nan. Skipping profile ...")
            return None
        binning = np.abs(np.nanmean(ddim))
        #print(binning)
        if binning == 0:
            print(f"Mean binning interval of {dim}, skipping profile.")
            return None

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

def bin_all_profiles(ds, vars, binning = None, agg='mean', dim='DEPTH',max_interval = 10):
    """
    Bin all profiles in the dataset along a specified dimension.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset with vertical or time-series profiles.
    vars : list of str
        Variables to be included in the binning.
    binning : float
        Bin size (in meters for DEPTH or seconds for TIME). If binning is None, the mean is taken in each profile.
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
    binned_df = grouped.apply(bin_profile, vars=all_vars, binning=binning, agg=agg, dim=dim, max_interval = max_interval)

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
    ### add the binning attribute, if None, then say the mean of each profile is taken
    ds_binned.attrs['binning'] = binning if binning is not None else 'mean of each profile'
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
    

def df_to_ds(mld_df, folder, glider_name):
    """
    Convert a pandas DataFrame containing mixed layer depth (MLD) data into an xarray Dataset.

    Parameters
    ----------
    mld_df : pd.DataFrame
        DataFrame containing MLD data with columns 'TIME', 'LONGITUDE', 'LATITUDE', and 'MLD'.

    Returns
    -------
    xr.Dataset
        An xarray Dataset with MLD data, indexed by 'TIME' and containing coordinates for 'LONGITUDE' and 'LATITUDE'.
    """
    # Add metadata
    mld_df['MISSION'] = folder
    mld_df['GLIDER'] = glider_name

    # Set TIME as index and convert to xarray.Dataset
    mld_df.set_index('TIME', inplace=True)
    mld_ds = xr.Dataset.from_dataframe(mld_df)

    mld_ds = mld_ds.sortby('TIME')
    ### somehow some time stamps are from 1970
    mld_ds = mld_ds.sel(TIME=slice('2006-01-01', '2009-12-31'))
    return mld_ds

variable_dict = {
    "PSAL": {
        "label": "Practical salinity",
        "units": "PSU",
        "colormap": cmo.haline
    },
    "TEMP": {
        "label": "Temperature",
        "units": "°C",
        "colormap": cmo.thermal
    },
    "THETA": {
        "label": "Potential temperature",
        "units": "°C",
        "colormap": cmo.thermal
    },
    "SIGMA_T": {
        "label": "Sigma-t",
        "units": "kg m⁻³",
        "colormap": cmo.dense
    },
    "SIGMA_1": {
        "label": "Sigma-1",
        "units": "kg m⁻³",
        "colormap": cmo.dense
    },
    "SIGTHETA": {
        "label": "Potential density (σθ)",
        "units": "kg m⁻³",
        "colormap": cmo.dense
    },
    "PRES": {
        "label": "Pressure",
        "units": "dbar",
        "colormap": cmo.deep
    },
    "DEPTH": {
        "label": "Depth",
        "units": "m",
        "colormap": cmo.deep
    },
    "GLIDER_VERT_VELO_MODEL": {
        "label": "Vertical glider velocity",
        "units": "cm s⁻¹",
        "colormap": cmo.delta
    },
    "GLIDER_HORZ_VELO_MODEL": {
        "label": "Horizontal glider velocity",
        "units": "cm s⁻¹",
        "colormap": cmo.delta
    },
    "VERTICAL_VELOCITY_MEASURED": {
        "label": "Measured vertical velocity",
        "units": "cm s⁻¹",
        "colormap": cmo.delta
    },
    "VERTICAL_WATER_VELOCITY": {
        "label": "Vertical water velocity",
        "units": "cm s⁻¹",
        "colormap": cmo.delta
    },
    "DISSIPATION_LEM_LOG": {
        "label": r"log$_{10}(e)$",
        "units": "W kg⁻¹",
        "colormap": cmo.delta
    },
    "VELOCITY_SCALE_2_LOG": {
        "label": r"log$_{10}(σ_{w}²)$",
        "units": "m² s⁻²",
        "colormap": cmo.delta
    },
    "SORTED_N2": {
        "label": "Adiabatically sorted buoyancy frequency (N²)",
        "units": "s⁻²",
        "colormap": cmo.dense
    },
    "SORTED_N_LOG": {
        "label": r"log$_{10}(N)$",
        "units": "s⁻¹",
        "colormap": cmo.delta
    },
    "SORTED_N2_LOG": {
        "label": r"log$_{10}(N²)$",
        "units": "s⁻²",
        "colormap": cmo.delta
    },
    "ADIABATIC_N2": {
        "label": "Adiabatic buoyancy frequency (N²)",
        "units": "s⁻²",
        "colormap": cmo.dense
    },
    "DISSIPATION_LEM_TOTAL": {
        "label": r"$\epsilon_{Gl}$",
        "units": "W m kg⁻¹",
        "colormap": cm.get_cmap('jet')
    },
    "EPSILON_TAU": {
        "label": r"$\epsilon_\tau$",
        "units": "W m kg⁻¹",
        "colormap": cmo.delta
    },
    "EPSILON_Q": {
        "label": r"$\epsilon_Q$",
        "units": "W m kg⁻¹",
        "colormap": cmo.delta
    },
    "MLD": {
        "label": "Mixed Layer Depth",
        "units": "m",
        "colormap": cmo.deep
    },
    "ALPHA_1": {
        "label": "Slope dV/dp",
        "units": "m³ kg⁻¹ Pa⁻¹",
        "colormap": cmo.thermal
    },
    "H_S": {
        "label": r"Stokes depth $h_s$",
        "units": "m",
        "colormap": cmo.deep
    },
    "slhf": {
        "label": "Net surface latent heat flux",
        "units": "W m⁻²",
        "colormap": cm.get_cmap('coolwarm')
    },
    "ssr": {
        "label": "Net surface short-wave radiation (solar)",
        "units": "W m⁻²",
        "colormap": cm.get_cmap('coolwarm')
    },
    "str": {
        "label": "Net surface long-wave radiation (thermal)",
        "units": "W m⁻²",
        "colormap": cm.get_cmap('coolwarm')
    },
    "sshf": {
        "label": "Net surface sensible heat flux",
        "units": "W m⁻²",
        "colormap": cm.get_cmap('viridis')
    },
    "u10": {
        "label": "10m u wind component",
        "units": "m/s",
        "colormap": cm.get_cmap('viridis')
    },
    "v10": {
        "label": "10m v wind component",
        "units": "m/s",
        "colormap": cm.get_cmap('viridis')
    },
}

def get_label(var: str):
    """
    Gets the label for a variable from the variable_dict dictionary.

    Parameters
    ----------
    var: str
        The variable (key) whose label is to be retrieved.

    Returns
    -------
    str: 
        The label corresponding to the variable `var`. If the variable is not found in `label_dict`,
        the function returns the variable name as the label.
    """
    if var in variable_dict:
        label = f'{variable_dict[var]["label"]}'
    else:
        label= f'{var}'
    return label

def get_unit(ds: xr.Dataset,var: str):
    """
    Gets the units for a variable from the dataset or the variable_dict dictionary.

    Parameters
    ----------
    ds: xarray.Dataset
        The dataset containing the variable `var`.
    var: str 
        The variable (key) whose units are to be retrieved.

    Returns
    -------
    str: 
        The units corresponding to the variable `var`. If the variable is found in `label_dict`,
        the associated units will be returned. If not, the function returns the units from `ds[var]`.
    """
    if var in variable_dict:
        return f'{variable_dict[var]["units"]}'
    elif 'units' in ds[var].attrs:
        return f'{ds[var].units}'
    else:
        return ""
    
def get_colormap(var: str):
    """
    Gets the colormap for a variable from the variable_dict dictionary. If the variable is not found,
    a default colormap (cmo.delta) is returned.

    Parameters
    ----------
    var: str
        The variable (key) whose colormap is to be retrieved.

    Returns
    -------
    colormap: matplotlib colormap or None
        The colormap corresponding to the variable `var`. If the variable is not found in `label_dict`,
        the function returns None.

    Notes
    -----
    Original Author: Till Moritz
    """
    if var in variable_dict:
        colormap = variable_dict[var]["colormap"]
    else:
        colormap = cmo.delta
    return colormap