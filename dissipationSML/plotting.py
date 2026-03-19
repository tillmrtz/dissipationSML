import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from cmocean import cm as cmo
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from matplotlib.dates import DateFormatter
from scipy import stats
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
from dissipationSML import tools, utilities
import importlib
importlib.reload(tools)
importlib.reload(utilities)
import matplotlib.cm as cm
from scipy.interpolate import interp1d
importlib.reload(utilities)
import matplotlib.ticker as mticker


import regionmask as rm
import ipywidgets as widgets
from IPython.display import display, clear_output


dir = os.path.dirname(os.path.realpath(__file__))
plotting_style = f"{dir}/plotting.mplstyle"
bathymetry = xr.open_dataset(f"{dir}/GEBCO_2024_IFR.nc")

def get_bathymetry_levels(bath, level_spacing=250):
        """
        This function computes the bathymetry levels for a given bathymetry dataset.

        Parameters
        ----------
        bath: xarray.Dataset
            Bathymetry dataset with 'elevation' variable.
        level_spacing: int, optional
            The spacing between contour levels. Default is 250 m.

        Returns
        -------
        levels: numpy.ndarray
            An array of bathymetry levels.
        contour_levels: numpy.ndarray
            An array of contour levels.
        max_level: int
            The maximum bathymetry level.
        """
        max_depth = np.max(-bath.elevation.values)  # Depths are negative
        max_level = level_spacing * (np.round(max_depth / level_spacing) + 1)
        levels = np.arange(0, max_level, level_spacing)
        contour_levels = levels[::2]  # Every second level
        return levels, contour_levels, max_level

def plot_glider_track(ds, mean_after="Profile", ax=None, **kw):
    """
    Plot glider track(s) on a map.

    If ds is a single Dataset: scatter is colored by TIME.
    If ds is a list of Datasets: each dataset is one mission, colored by mission.

    Parameters
    ----------
    ds : xarray.Dataset or list of xarray.Dataset
        Dataset(s) with LATITUDE, LONGITUDE, TIME, GLIDER, MISSION.
    mean_after : {"Profile", "Cast", "None"}, default="Profile"
        Whether to average lat/lon/time per profile, per cast, or use raw values.
    ax : matplotlib.axes.Axes, optional
        Axis to plot on. If None, a new figure/axis is created.
    **kw : dict
        Extra keyword arguments passed to ax.scatter.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """

    map_kw = ccrs.PlateCarree()

    if ax is None:
        fig, ax = plt.subplots(subplot_kw={"projection": map_kw}, figsize=(12, 8))
    else:
        fig = ax.get_figure()
        # If provided axis has no Cartopy projection, replace it
        if not hasattr(ax, 'projection'):
            # Remember position and remove old one
            pos = ax.get_position()
            ax.remove()
            ax = fig.add_axes(pos, projection=ccrs.PlateCarree())

    # Define bounding box
    lon_min, lon_max = -15, -6
    lat_min, lat_max = 60, 65

    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=map_kw)
    ax.set_aspect("auto")

    # Bathymetry (assuming bathymetry is global variable)
    bath = bathymetry.sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max))
    levels, contour_levels, max_level = get_bathymetry_levels(bath)

    cmap_bath = plt.get_cmap("Blues", len(levels))

    pcm = ax.pcolormesh(
        bath.lon, bath.lat, abs(bath.elevation.values),
        cmap=cmap_bath, vmin=0, vmax=max_level, transform=map_kw
    )
    ax.contour(
        bath.lon, bath.lat, abs(bath.elevation.values),
        levels=contour_levels, colors="black", linewidths=0.5, transform=map_kw
    )

    # Single dataset → color by TIME
    if isinstance(ds, xr.Dataset):
        if mean_after == "Profile":
            latitudes = ds.LATITUDE.groupby(ds.PROFILE_NUMBER).mean().values
            longitudes = ds.LONGITUDE.groupby(ds.PROFILE_NUMBER).mean().values
            times = ds.TIME.groupby(ds.PROFILE_NUMBER).mean().values
        elif mean_after == "Cast":
            latitudes = ds.LATITUDE.groupby(ds.CAST).mean().values
            longitudes = ds.LONGITUDE.groupby(ds.CAST).mean().values
            times = ds.TIME.groupby(ds.CAST).mean().values
        else:
            latitudes = ds.LATITUDE.values
            longitudes = ds.LONGITUDE.values
            times = ds.TIME.values

        sc = ax.scatter(longitudes, latitudes, c=times, cmap="inferno", s=10, marker="o", **kw)

        # Time colorbar
        cbar = plt.colorbar(sc, ax=ax, pad=0.01, shrink=1)
        cbar.ax.set_yticklabels([pd.to_datetime(t).strftime("%Y-%b-%d") for t in cbar.get_ticks()])

    # List of datasets → color by mission
    elif isinstance(ds, list):
        def mission_label(dsi):
            return f"{dsi['GLIDER'].values[0]}/{dsi['MISSION'].values[0]}"

        # Build (label, date) pairs
        mission_info = []
        for dsi in ds:
            label = mission_label(dsi)
            date = pd.to_datetime(dsi.TIME.min().values)  # mission start time
            mission_info.append((label, date))

        # Sort missions by date
        mission_info = sorted(mission_info, key=lambda x: x[1])
        mission_labels = [lab for lab, _ in mission_info]

        # Colormap
        mission_cmap = cm.get_cmap("inferno", len(mission_labels))
        mission_color_dict = {lab: mission_cmap(i) for i, lab in enumerate(mission_labels)}

        # Plot each mission
        for dsi, (label, _) in zip(ds, mission_info):
            if mean_after == "Profile":
                latitudes = dsi.LATITUDE.groupby(dsi.PROFILE_NUMBER).mean().values
                longitudes = dsi.LONGITUDE.groupby(dsi.PROFILE_NUMBER).mean().values
            elif mean_after == "Cast":
                latitudes = dsi.LATITUDE.groupby(dsi.CAST).mean().values
                longitudes = dsi.LONGITUDE.groupby(dsi.CAST).mean().values
            else:
                latitudes = dsi.LATITUDE.values
                longitudes = dsi.LONGITUDE.values

            ax.scatter(
                longitudes, latitudes,
                color=mission_color_dict[label], s=10, marker="o", label=label, **kw
            )

        # Colorbar with mission order by date
        norm = mcolors.BoundaryNorm(np.arange(len(mission_labels) + 1) - 0.5, mission_cmap.N)
        sm = plt.cm.ScalarMappable(cmap=mission_cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(
            sm, ax=ax, pad=0.01, shrink=1, ticks=np.arange(len(mission_labels))
        )
        cbar.ax.set_yticklabels(mission_labels)

    # Bathymetry colorbar
    cbar_bath = plt.colorbar(pcm, ax=ax, label="Depth (m)", pad=0.01, shrink=1)
    cbar_bath.set_ticks(levels)

    # Features
    ax.add_feature(cfeature.LAND, color="lightgray", zorder=10)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, zorder=11)

    # Labels & title
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Glider Track with Bathymetry")

    # Gridlines with custom ticks
    gl = ax.gridlines(draw_labels=True, color="black", alpha=0.5, linestyle="--")
    gl.xlocator = mticker.FixedLocator([-14, -12, -10, -8, -6])
    gl.top_labels = False
    gl.right_labels = False

    return fig, ax


def plot_profile(ds: xr.Dataset, profile_num: int, vars: list = ['TEMP','PSAL','SIGMA_T'], use_bins: bool = False, binning: float = 2,ax = None) -> tuple:
    """
    Plots binned temperature, salinity, and density against depth on a single plot with three x-axes.

    Parameters
    ----------
    ds: xarray.Dataset
        Xarray dataset in OG1 format with at least PROFILE_NUMBER, DEPTH, TEMPERATURE, SALINITY, and DENSITY.
    profile_num: int
        The profile number to plot.
    vars: list
        The variables to plot. Default is ['TEMP','PSAL','DENSITY'].
    binning: int
        The depth resolution for binning.
    use_bins: bool
        If True, use binned data instead of raw data.

    Returns
    -------
    fig: matplotlib.figure.Figure
        The figure object containing the plot.
    ax1: matplotlib.axes.Axes
        The axis object containing the primary plot.

    Notes
    -----
    Original Author: Till Moritz
    """
    # Remove empty strings from vars
    vars = [v for v in vars if v] 
    # If vars is empty, show an empty plot
    if not vars:
        if ax is None:  
            fig, ax1 = plt.subplots(figsize=(12, 9))
            force_plot = True
        else:
            fig = plt.gcf()
            force_plot = False
        ax1.set_title(f'Profile {profile_num} (No Variables Selected)')
        ax1.set_ylabel('Depth (m)')
        ax1.invert_yaxis()
        ax1.grid(True)
        return fig, ax1
    
    if len(vars) > 3:
        raise ValueError("Only three variables can be plotted at once, chose less variables")
    
    with plt.style.context(plotting_style):
        if ax is None:  
            fig, ax1 = plt.subplots(figsize=(12, 9))   
            force_plot = True
        else:
            fig = plt.gcf()
            force_plot = False
            ax1 = ax  # Use the first axis if provided

        profile = ds.where(ds.PROFILE_NUMBER == profile_num, drop=True)
        if use_bins:
            profile = utilities.bin_profile(profile, vars, binning)

        # Plot binned data
        mission = ds.id.split('_')[1][0:8]
        glider = ds.id.split('_')[0]

        axs = [ax1, ax1.twiny(), ax1.twiny()]
        colors = ['red', 'blue', 'grey']
        s = 10 + binning

        for i, var in enumerate(vars):
            ax = axs[i]
            unit = utilities.get_unit(ds, var)
            label = utilities.get_label(var)
            ax.plot(profile[var], profile['DEPTH'], color=colors[i], label=label)
            ax.scatter(profile[var], profile['DEPTH'], color=colors[i], marker='o', s=s)
            ax.set_xlabel(f'{label} [{unit}]', color=colors[i])
            ax.tick_params(axis='x', colors=colors[i])
            ax.spines['top'].set_visible(False)
            if i > 0:
                ax.xaxis.set_ticks_position('bottom')
                ax.spines['bottom'].set_position(('axes', -0.09*i))
            ax.xaxis.set_label_coords(0.5, -0.05-0.105*i)

        # Set pressure as y-axis (Increasing Downward)
        ax1.grid(True)
        ax1.set_ylabel('Depth (m)')
        ax1.invert_yaxis()  # Pressure increases downward
        ax1.set_title(f'Profile {profile_num} ({glider} on mission: {mission})')

    return fig, ax1

def plot_CR(ds: xr.Dataset, profile_num: int, use_bins: bool = False, binning: float = 2,ax = None) -> tuple:
    """
    Plots the convective resistance (CR) of a profile against depth based on calculate_CR_for_all_depth function.
    For the calculation, the density anomaly with reference to 1000 kg/m3 is used ('SIGMA_1')

    Parameters
    ----------
    ds: xarray.Dataset
        Xarray dataset in OG1 format with at least PROFILE_NUMBER, DEPTH, SIGMA_1.
    profile_num: int
        The profile number to plot.
    use_bins: bool
        If True, use binned data instead of raw data.
    binning: int
        The depth resolution for binning.

    Returns
    -------
    fig: matplotlib.figure.Figure
        The figure object containing the plot.
    ax: matplotlib.axes.Axes
        The axis object containing the primary plot.

    Notes
    -----
    Original Author: Till Moritz
    """
    vars = ['SIGMA_1', 'DEPTH']

    profile = ds.where(ds.PROFILE_NUMBER == profile_num, drop=True)

    if use_bins:
        profile = utilities.bin_profile(profile, vars, binning)

    CR_df = tools.calculate_CR_for_all_depth(profile)

    depth = CR_df['DEPTH'].values
    CR = CR_df['CR'].values

    with plt.style.context(plotting_style):
        if ax is None:  
            fig, ax = plt.subplots(figsize=(12, 9)) 
            force_plot = True
        else:
            fig = plt.gcf()
            force_plot = False
        ax.plot(CR, depth, label='CR')
        ax.scatter(CR, depth, marker='o', s=10+binning)
        ax.set_xlabel('Convective Resistance (CR)')
        ax.set_ylabel('Depth (m)')
        ax.invert_yaxis()
        ax.set_title(f'Profile {profile_num} (Convective Resistance)')
        ax.grid(True)
        ax.legend()
    
    return fig, ax

def plot_scatter(ds, vars=['PSAL', 'TEMP', 'DENSITY'], start=None, end=None, mld_df=None):
    """
    Plots scatter plots of the dataset with TIME on the x-axis and DEPTH on the y-axis.
    Colors indicate variable values. Optionally overlays Mixed Layer Depth (MLD) data.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset with at least 'TIME', 'DEPTH', and the requested variables.
    vars : list of str
        Variables to plot.
    start : str
        Start time in 'YYYY-MM-DD' or 'YYYY-MM-DDTHH:MM:SS' format.
    end : str
        End time in 'YYYY-MM-DD' or 'YYYY-MM-DDTHH:MM:SS' format.
    mld_df : pandas.DataFrame, optional
        DataFrame containing MLD values with 'TIME' and 'MLD' columns.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : list of matplotlib.axes._subplots.AxesSubplot
    """

    if start is not None or end is not None:
        if start is None:
            start = ds.TIME.min().values
        if end is None:
            end = ds.TIME.max().values
        
        mask = (ds.TIME >= np.datetime64(start)) & (ds.TIME <= np.datetime64(end))
        
        dim = list(ds.dims.keys())[0]
        if dim == 'N_MEASUREMENTS':
            ds = ds.sel(N_MEASUREMENTS=mask)
        elif dim == 'TIME':
            ds = ds.sel(TIME=mask)
        
        if mld_df is not None:
            mld_df = mld_df[(mld_df['PROFILE_NUMBER'] >= start) & (mld_df['PROFILE_NUMBER'] <= end)]

    num_vars = len(vars)
    fig, ax = plt.subplots(num_vars, 1, figsize=(20, 7 * num_vars), sharex=True,
                           gridspec_kw={'height_ratios': [8] * num_vars})
    if num_vars == 1:
        ax = [ax]

    x_data = mdates.date2num(ds.TIME.values)
    depth = ds['DEPTH'].values
    has_density = any(utilities.get_colormap(var) == cmo.dense for var in vars)
    s = 5

    for i, var in enumerate(vars):
        if var not in ds:
            raise ValueError(f'Variable "{var}" not found in dataset.')

        values = ds[var].values
        mask = np.isnan(values) | np.isnan(depth)

        cmap = utilities.get_colormap(var)
        if cmap == cmo.delta and np.any(values < 0) and np.any(values > 0):
            norm = mcolors.TwoSlopeNorm(vmin=np.nanpercentile(values, 0.5), vcenter=0, vmax=np.nanpercentile(values, 99.5))
        else:
            norm = None

        scatter = ax[i].scatter(
            x_data[~mask], depth[~mask], c=values[~mask], s=s, cmap=cmap, norm=norm,
            vmin=None if norm else np.nanpercentile(values, 0.5),
            vmax=None if norm else np.nanpercentile(values, 99.5)
        )

        if mld_df is not None and (has_density and cmap == cmo.dense or not has_density and i == 0):
            ax[i].plot(mdates.date2num(mld_df['TIME']), mld_df['MLD'], 'ko-', linewidth=1, markersize=2, label='MLD')
            ax[i].legend(loc='upper left', fontsize=8)

        unit = utilities.get_unit(ds, var)
        label = utilities.get_label(var)

        ax[i].invert_yaxis()
        ax[i].set_ylabel('Depth (m)')
        ax[i].grid()
        ax[i].set_title(f'Scatter plot of {label}')

        cbar = plt.colorbar(scatter, ax=ax[i], pad=0.03)
        cbar.set_label(f'{label} [{unit}]', labelpad=20, rotation=270)

    ax[-1].xaxis.set_major_locator(mdates.AutoDateLocator())
    delta = x_data[-1] - x_data[0] if len(x_data) > 1 else 1
    date_format = '%Y-%m-%d%H:%M' if delta < 2 else '%Y-%m-%d'
    ax[-1].xaxis.set_major_formatter(mdates.DateFormatter(date_format))
    plt.xticks(rotation=45)
    ax[-1].set_xlabel('Time')

    plt.tight_layout()
    plt.show()

    return fig, ax

def plot_section(ds, vars=['PSAL', 'TEMP', 'DENSITY'], v_res=2, start=None, end=None, mld_df = None, levels=None, ax = None, log_scale = False):
    """
    Plots a section of the dataset with PROFILE_NUMBER on the x-axis, DEPTH on the y-axis,
    and mean TIME per profile as secondary x-axis (automatically spaced).

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset with at least PROFILE_NUMBER, TIME, DEPTH, and target variables.
    vars : str or list of str
        Variables to visualize. If a single variable is provided, it will be converted to a list.
    v_res : float
        Vertical resolution (DEPTH binning).
    start : int or None
        Start PROFILE_NUMBER (inclusive).
    end : int or None
        End PROFILE_NUMBER (inclusive).
    mld_df : pd.DataFrame
        MLD as a pandas Dataframe, which is the result of the MLD calculation compute_mld(). The dataframe should contain the profile number, MLD and the mean time profile.
    levels : None, bool, or list/array
        - None (default): continuous colormap (pcolormesh).
        - True: use 10 equally spaced discrete levels (rounded to 1 decimal).
        - list/array of floats: use exactly these levels (non-uniform spacing honored).
    log_scale : bool
        If True, use logarithmic scale for color mapping.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : list of matplotlib.axes.Axes

    Notes
    -----
    Original Author: Till Moritz
    """
    if not isinstance(vars, list):
        vars = [vars]

    if start is not None or end is not None:
        if start is None:
            start = ds.PROFILE_NUMBER.min().values
        if end is None:
            end = ds.PROFILE_NUMBER.max().values
        
        mask = (ds.PROFILE_NUMBER >= start) & (ds.PROFILE_NUMBER <= end)
        
        dim = list(ds.dims.keys())[0]
        if dim == 'N_MEASUREMENTS':
            ds = ds.sel(N_MEASUREMENTS=mask)
        elif dim == 'TIME':
            ds = ds.sel(TIME=mask)
        
        if mld_df is not None:
            mld_df = mld_df[(mld_df['PROFILE_NUMBER'] >= start) & (mld_df['PROFILE_NUMBER'] <= end)]

    with plt.style.context(plotting_style):
        num_vars = len(vars)

        # --- Handle provided axes ---
        if ax is not None:
            # Ensure ax is iterable
            if not isinstance(ax, (list, np.ndarray)):
                ax = [ax]
            if len(ax) != num_vars:
                raise ValueError(f"Number of provided axes ({len(ax)}) does not match number of variables ({num_vars}).")
            fig = ax[0].get_figure()
        else:
            # Create new figure and axes if none provided
            fig, ax = plt.subplots(
                num_vars, 1,
                figsize=(15, 5 * num_vars),
                sharex=True,
                gridspec_kw={'height_ratios': [8] * num_vars}
            )
            if num_vars == 1:
                ax = [ax]

        images = []


        x_plot = ds['PROFILE_NUMBER'].values

        has_density_plot = any(utilities.get_colormap(var) == cmo.dense for var in vars)

        # Compute mean time per profile
        df_time = ds[['TIME', 'PROFILE_NUMBER']].to_dataframe().dropna()
        if df_time.index.name == 'TIME':
            df_time = df_time.reset_index()
        mean_times = df_time.groupby('PROFILE_NUMBER')['TIME'].mean()

        for i, var in enumerate(vars):
            if var not in ds:
                raise ValueError(f'Variable "{var}" not found in dataset.')

            values = ds[var].values
            depth = ds['DEPTH'].values

            p = 1
            z = v_res

            varG, profG, depthG = utilities.construct_2dgrid(x_plot, depth, values, p, z, x_bin_center=False)

            cmap = utilities.get_colormap(var)

            vmin = np.nanpercentile(values, 0.5)
            vmax = np.nanpercentile(values, 99.5)

            # --- Levels handling and validation ---
            levs = None
            if isinstance(levels, (list, np.ndarray)):
                levs = np.asarray(levels, dtype=float)
                levs = np.unique(np.sort(levs))   # sort & remove duplicates
                if levs.size < 2:
                    raise ValueError("levels must contain at least two distinct values.")
            elif levels is True:
                # 10 evenly spaced rounded levels (one decimal)
                if np.isfinite(vmin) and np.isfinite(vmax) and (vmax > vmin):
                    if log_scale == True:
                        levs = np.linspace(np.log10(vmin), np.log10(vmax), 10)
                        levs = 10 ** np.round(levs, 2)  # round in log space
                    else:
                        levs = np.round(np.linspace(vmin, vmax, 10), 2)
                        levs = np.unique(levs)
                    if levs.size < 2:
                        # fallback if rounding collapsed values
                        levs = np.array([vmin, vmax])
                else:
                    levs = None
            else:
                levs = None

            # --- Plot either discrete contourf (levels) or continuous pcolormesh ---
            if levs is not None:
                # contourf will use levs as numeric boundaries; do not use BoundaryNorm
                cf = ax[i].contourf(profG, depthG, varG, levels=levs, cmap=cmap, extend="both")
                # optional contour lines
                #ax[i].contour(profG, depthG, varG, levels=levs, colors="k", linewidths=0.3, alpha=0.5)
                mappable = cf
            else:
                if log_scale:
                    im = ax[i].pcolormesh(profG, depthG, varG, cmap=cmap, norm=LogNorm(vmin=vmin, vmax=vmax))
                else:
                    im = ax[i].pcolormesh(profG, depthG, varG, cmap=cmap, vmin=vmin, vmax=vmax)
                mappable = im

            if mld_df is not None:
                if (has_density_plot and cmap == cmo.dense) or (not has_density_plot and i == 0):
                    ax[i].plot(mld_df['PROFILE_NUMBER'], mld_df['MLD'], color='black', marker='o', linewidth=0.5,
                            label='Mixed Layer Depth', markersize=2)
                    #ax[i].legend(loc='lower right', fontsize=8)

            unit = utilities.get_unit(ds, var)
            label = utilities.get_label(var)

            total_profiles = x_plot[-1] - x_plot[0]
            ax[i].invert_yaxis()
            ax[i].set_ylabel('Depth (m)')
            ax[i].grid(True)
            ax[i].set_title(f'Section plot of {label}')
            ax[i].set_xlim(np.min(x_plot)-total_profiles/50, np.max(x_plot)+total_profiles/50)

            
            # --- Colorbar: make spacing proportional when levs provided ---
            if levs is not None:
                # Provide boundaries=levs and spacing='proportional' so lengths reflect actual numeric gaps
                cbar = plt.colorbar(mappable, ax=ax[i], pad=0.03, boundaries=levs, spacing='proportional')
                # ticks at the boundary levels (format nicely with 1 decimal)
                cbar.set_ticks(levs)
                cbar.set_ticklabels([f"{l:.2f}" for l in levs])
            else:
                if log_scale:
                    cbar = plt.colorbar(mappable, ax=ax[i], pad=0.03)
                    cbar.ax.set_yscale('log')
                else:
                    cbar = plt.colorbar(mappable, ax=ax[i], pad=0.03)
            cbar.set_label(f'{label} [{unit}]', labelpad=20, rotation=270)

        # Main x-axis: profile numbers
        ax[-1].set_xlabel('Profile Number')

        # Get mean time per profile (datetime) and profile numbers
        times = pd.to_datetime(mean_times)
        profiles = mean_times.index.values
        time_nums = mdates.date2num(times)  # matplotlib float format for dates

        # Build interpolators
        to_time = interp1d(profiles, time_nums, bounds_error=False, fill_value="extrapolate")
        to_profile = interp1d(time_nums, profiles, bounds_error=False, fill_value="extrapolate")

        # Create a transform that maps profile numbers → time for the secondary x-axis
        def forward(x):
            return to_time(x)

        def inverse(x):
            return to_profile(x)

        # Create the secondary axis (top), linked to the bottom profile axis
        time_ax = ax[-1].secondary_xaxis("bottom", functions=(forward, inverse))
        #time_ax.set_xlabel("Mean Time per Profile")
        time_ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        time_delta = time_nums[-1] - time_nums[0]
        if time_delta < 5:
            time_ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        else:
            time_ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        
        time_ax.spines['bottom'].set_position(('outward', 40))
        time_ax.tick_params(rotation=35)
    
    #plt.tight_layout()
    #plt.show()
    
    return fig, ax


def plot_vertical_resolution(ds: xr.Dataset, profile_num: int) -> tuple:
    """
    Plots a histogram of the vertical distances between consecutive measurements.

    Parameters
    ----------
    ds: xarray.Dataset
        Xarray dataset in OG1 format with at least PRESSURE
    profile_num: int
        The profile number to plot.

    Returns
    -------
    fig: matplotlib.figure.Figure
        The figure object containing the plot.
    ax: matplotlib.axes.Axes
        The axis object containing the primary plot.
    """
    with plt.style.context(plotting_style):  # Assuming `plotting_style` is defined elsewhere
        fig, ax = plt.subplots(figsize=(8, 6))

        # Select the specific profile
        profile = ds.where(ds.PROFILE_NUMBER == profile_num, drop=True)
        depth = profile.DEPTH.values
        distances = np.abs(np.diff(depth))

        if np.abs(distances-distances[0]).max() < 0.01:  # all distances are the same
            width = 0.5
            bins = [distances[0]-width/2, distances[0]+width/2]
            x_lim = [0, 2*distances[0]]
        else:
            bins = 20  # default bin count
            x_lim = [0, np.nanmax(distances)*1.2]

        # Plot histogram of vertical distances
        ax.hist(distances, bins=bins, color='blue', alpha=0.7)#,density=True)#,stacked=True)
        ax.set_xlabel('Vertical Distance (m)')
        ax.set_ylabel('Number of Measurements')
        ax.set_title(f'Profile {profile_num} Vertical Resolution')

    return fig, ax

def plot_time_resolution(ds: xr.Dataset, profile_num: int) -> tuple:
    """
    Plots a histogram of the time differences between consecutive measurements.

    Parameters
    ----------
    ds: xarray.Dataset
        Xarray dataset in OG1 format with at least TIME
    profile_num: int
        The profile number to plot.

    Returns
    -------
    fig: matplotlib.figure.Figure
        The figure object containing the plot.
    ax: matplotlib.axes.Axes
        The axis object containing the primary plot.
    """
    with plt.style.context(plotting_style):  # Assuming `plotting_style` is defined elsewhere
        fig, ax = plt.subplots(figsize=(8, 6))

        # Select the specific profile
        profile = ds.where(ds.PROFILE_NUMBER == profile_num, drop=True)
        depth = profile.DEPTH.values
        depth_diff = np.diff(depth)
        time = profile.TIME.values.astype('float64')
        time_diff = np.diff(time) * 1e-9
        ### take out all time_diff values that are larger than 30 seconds and print them
        print('Time differences larger than 90 seconds:', time_diff[time_diff >= 91] , 'at depths:', (depth[1:][time_diff >= 91]+depth[:-1][time_diff >= 91])/2)
        time_diff = time_diff[time_diff <= 91]
        # Plot histogram of time differences
        ax.hist(time_diff, bins=20, color='blue', alpha=0.7)
        ax.set_xlabel('Time Difference (s)')
        ax.set_ylabel('Number of Measurements')
        ax.set_title(f'Profile {profile_num} Time Resolution')

    return fig, ax

def plot_min_max_depth(ds: xr.Dataset, bins= 20, ax = None, **kw: dict):
    """
    This function can be used to plot the maximum depth of each profile in a dataset.
    
    Parameters
    ----------
    ds: xarray on OG1 format containing the profile number and the maximum depth. 
    bins: int, optional (default=20)
    
    Returns
    -------
    fig: matplotlib.figure.Figure
        The figure object containing the plot.
    ax: matplotlib.axes.Axes
        The axis object containing the primary plot.
    """
    min_depths, max_depths = tools.min_max_depth_per_profile(ds)
    with plt.style.context(plotting_style):
        if ax is None:  
            fig, ax = plt.subplots(1, 2)  
            force_plot = True
        else:
            fig = plt.gcf()
            force_plot = False
            
        ax[0].hist(min_depths, bins=bins)
        ax[0].set_xlabel(f'Min depth ({min_depths.units})')
        ax[0].set_ylabel('Number of profiles')
        ax[0].set_title('Min depth per profile')
        ax[1].hist(max_depths, bins=bins)
        ax[1].set_xlabel(f'Max depth ({max_depths.units})')
        ax[1].set_ylabel('Number of profiles')
        ax[1].set_title('Max depth per profile')
        [a.grid() for a in ax]
    return fig, ax

def plot_IFR_region_on_map(IFR_region: rm.Regions):
    """
    This function plots the glider track on a map, with latitude and longitude colored by time.

    Parameters
    ----------
    ds: xarray in OG1 format with at least TIME, LATITUDE, and LONGITUDE.
    ax: Optional; axis to plot the data.
    kw: Optional; additional keyword arguments for the scatter plot.

    Returns
    -------
    One plot with the map of the glider track.
    fig: matplotlib.figure.Figure
    ax: matplotlib.axes._subplots.AxesSubplot
    """
   
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
    lon_lat_range = [-16, -6, 58, 67]

    # plot the IFR region with numbers at each corner
    IFR_region.plot(ax=ax, add_label=False, line_kws={'color': 'red'})

    # Extract corner coordinates
    coords = IFR_region[0].coords  # Get the first polygon's exterior (first 4 points)

    # Add corner numbers
    for i, (lon, lat) in enumerate(coords, start=1):  # Start numbering from 1
        ax.text(lon, lat, str(i), fontsize=12, color='black', weight='bold',
                ha='center', va='center', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

    ax.set_extent(lon_lat_range, crs=ccrs.PlateCarree())

    # Add map features
    ax.add_feature(cfeature.LAND, color='lightgray')
    ax.add_feature(cfeature.OCEAN, color='lightblue')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)

    # Extract bathymetry data in the specified range
    bath = bathymetry.sel(lon=slice(lon_lat_range[0], lon_lat_range[1]), 
                            lat=slice(lon_lat_range[2], lon_lat_range[3]))

    max_depth = np.min(bath.elevation.values)  # Depths are negative
    levels = np.linspace(0, max_depth, 7)  # Generate 8 levels
    rounded_levels = np.round(levels / 10) * 10  # Round levels to nearest 10

    # Plot bathymetry contours
    contours = bath.elevation.plot.contour(ax=ax, transform=ccrs.PlateCarree(), 
                                               levels=rounded_levels, colors='black', linewidths=0.5)
    # Add contour labels as positive values
    ax.clabel(contours,fmt='%d m', fontsize=6, colors='black')
    ### plot the bathymetry as a color map
    #cmap = plt.get_cmap('viridis')
    #bath.elevation.plot(ax=ax, transform=ccrs.PlateCarree(), cmap=cmap, add_colorbar=True)
    ## add colorbar

    ax.set_xlabel(f'Longitude')
    ax.set_ylabel(f'Latitude')
    ax.set_title('Glider Track')
    gl = ax.gridlines(draw_labels=True, color='black', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False

    return fig, ax

def plot_dive_depth(ds, dive_number):
    """
    This function plots the depth of one dive against time.
    
    Parameters
    ----------
    ds: xarray.Dataset
        Xarray dataset in OG1 format with at least TIME, DEPTH, and DIVE_NUMBER.
    dive_number: int
        The dive number to plot.

    Returns
    -------
    fig: matplotlib.figure.Figure
        The figure object containing the plot.
    ax: matplotlib.axes.Axes
        The axis object containing the primary plot.
    """
    with plt.style.context(plotting_style):
        fig, ax = plt.subplots(figsize=(12, 6))

        # Select the specific dive
        dive = ds.sel(N_MEASUREMENTS = ds.DIVE_NUMBER == dive_number, drop=True)
        time = dive.TIME.values
        depth = dive.DEPTH.values

        # Plot depth against time
        dive.plot.scatter(x='TIME', y='DEPTH', ax=ax, color='blue', s=10)
        ax.invert_yaxis()
        ax.set_xlabel('Time')
        ax.set_ylabel('Depth (m)')
        ax.set_title(f'Dive {dive_number} Depth Profile')
        ### only plot the time at the x-ticks and the date at the x-labels
        ax.grid(True)
    return fig, ax

def get_var_styles(var_list):
    """Return line, marker, and color styles for a list of variables."""
    linestyles = ['-', '--', '-.', ':']
    markers = ['o', '*', 'x', '^']
    colors = ['tab:blue', 'tab:orange', 'tab:gray', 'tab:cyan']
    return (
        {var: linestyles[i % len(linestyles)] for i, var in enumerate(var_list)},
        {var: markers[i % len(markers)] for i, var in enumerate(var_list)},
        {var: colors[i % len(colors)] for i, var in enumerate(var_list)}
    )


def plot_var_from_mld(mld_ds, vars, years=None, rolling_str='12h',
                      plot_type='both', mission_cbar=True, one_plot=False):
    """
    Plot one or more variables from MLD dataset(s), colored by mission or variable.

    Parameters
    ----------
    mld_ds : xarray.Dataset or list of xarray.Dataset
        MLD dataset(s) containing the variable(s) to plot.
    vars : str or list of str
        The variable(s) name(s) to plot.
    years : list of int or None
        The years to plot. If None, plot full time range.
    rolling_str : str
        Rolling mean window string (e.g., '12h').
    plot_type : str
        What to plot: 'both', 'scatter', or 'rolling'.
    mission_cbar : bool
        Whether to display the mission colorbar.
    one_plot : bool
        If True, all variables go into one plot per year.

    Returns
    -------
    fig, axes : matplotlib Figure and Axes
    """

    if plot_type not in {'both', 'scatter', 'rolling'}:
        raise ValueError("plot_type must be 'both', 'scatter', or 'rolling'.")

    def get_mission_label(ds):
        return f"{ds['GLIDER'].values[0]}/{ds['MISSION'].values[0]}"

    # Normalize inputs
    mld_ds = [mld_ds] if not isinstance(mld_ds, list) else mld_ds
    var_list = [vars] if isinstance(vars, str) else vars
    years = [years] if isinstance(years, int) else years

    # Get styles
    var_linestyle, var_marker, var_color_dict = get_var_styles(var_list)

    # Color per mission
    mission_labels = [get_mission_label(ds) for ds in mld_ds]
    unique_missions = sorted(set(mission_labels), key=lambda x: x.split('/')[1])
    mission_cmap = cm.get_cmap('tab20', len(unique_missions))
    mission_color_dict = {label: mission_cmap(i) for i, label in enumerate(unique_missions)}

    # Determine number of subplots
    time_splits = years if years else [None]
    num_plots = len(time_splits) if one_plot else len(time_splits) * len(var_list)

    # Setup figure
    with plt.style.context(plotting_style):
        figsize = (25, 8 * num_plots + 2) if mission_cbar else (25, 8 * num_plots)
        fig, axes = plt.subplots(num_plots, 1, figsize=figsize, squeeze=False)
        axes = axes.flatten()

        # Determine time range
        all_times = np.concatenate([ds['TIME'].values for ds in mld_ds])
        full_range = (all_times.min(), all_times.max())

        plot_idx = 0
        for year in time_splits:
            for v_block in ([var_list] if one_plot else [[v] for v in var_list]):
                ax = axes[plot_idx]
                label_set = set()

                # Title
                year_str = f"{year}" if year else "Full Range"
                ax.set_title(f"{' , '.join(v_block)} colored by "
                             f"{'Mission' if mission_cbar else 'Variable'} - {year_str}", fontsize=20)

                unit_set = set()

                for ds in mld_ds:
                    ds = ds.sortby('TIME')
                    mission_label = get_mission_label(ds)

                    # Filter by year
                    if year:
                        ds = ds.sel(TIME=slice(f'{year}-01-01', f'{year}-12-31'))
                        if ds.TIME.size == 0:
                            continue

                    rolling = ds.resample(TIME=rolling_str, origin='epoch').mean(dim='TIME').sortby('TIME')
                    delta = np.timedelta64(int(rolling_str[:-1]), rolling_str[-1])
                    rolling['TIME'] = rolling['TIME'] + delta / 2
                    
                    if year:
                        rolling = rolling.sel(TIME=slice(f'{year}-01-01', f'{year}-12-31'))

                    for var in v_block:
                        if var not in ds:
                            continue

                        label = utilities.get_label(var)
                        unit = utilities.get_unit(ds, var)
                        unit_set.add(unit)

                        color = mission_color_dict[mission_label] if mission_cbar else var_color_dict[var]
                        linestyle = var_linestyle[var]
                        marker = var_marker[var]

                        scatter_lbl = f"{label} (scatter)"
                        roll_lbl = f"{label} (rolling)"

                        if plot_type in {'both', 'scatter'}:
                            lbl = scatter_lbl if scatter_lbl not in label_set else None
                            ax.scatter(ds['TIME'].values, ds[var].values, color=color,
                                       marker=marker, s=35, alpha=0.6, label=lbl)
                            label_set.add(scatter_lbl)

                        if plot_type in {'both', 'rolling'}:
                            lbl = roll_lbl if roll_lbl not in label_set else None
                            ax.plot(rolling['TIME'].values, rolling[var].values, color=color,
                                    linestyle=linestyle, linewidth=2, label=lbl)
                            label_set.add(roll_lbl)

                ax.set_xlim(full_range if not year else
                            (np.datetime64(f'{year}-01-01'), np.datetime64(f'{year}-12-31')))

                if len(unit_set) == 1:
                    labels = " , ".join([f"{utilities.get_label(v)}" for v in v_block])
                    ax.set_ylabel(f"{labels} [{list(unit_set)[0]}]", fontsize=18)
                else:
                    ylabel = " , ".join([f"{utilities.get_label(v)} [{utilities.get_unit(ds, v)}]" for v in v_block])
                    ax.set_ylabel(ylabel, fontsize=18)

                ax.grid(True)
                ax.legend(fontsize=12)
                plot_idx += 1

        axes[-1].set_xlabel('Time', fontsize=18)
        fig.tight_layout(rect=[0, 0, 1, 1])

        if mission_cbar:
            sm = cm.ScalarMappable(norm=mcolors.Normalize(vmin=0, vmax=len(unique_missions) - 1),
                                   cmap=mission_cmap)
            cbar = fig.colorbar(sm, ax=axes, orientation='horizontal',
                                fraction=0.15, pad=0.1 / num_plots)
            cbar.set_ticks(np.arange(len(unique_missions)))
            cbar.set_ticklabels(unique_missions)
            cbar.set_label('Mission (Glider/MissionDate)', fontsize=14)

    return fig, axes

def fit_linear_regression(x, y):
    """
    Fits a linear regression to the provided x and y data.

    Parameters
    ----------
    x : array-like
        Independent variable data.
    y : array-like
        Dependent variable data.

    Returns
    -------
    slope : float
        The slope of the fitted line.
    intercept : float
        The intercept of the fitted line.
    residuals : float or None
        The sum of squared residuals, or None if no residuals are available.
    """
    msk = ~np.isnan(x) & ~np.isnan(y)
    x = x[msk]
    y = y[msk]
    p = np.polyfit(x, y, 1)
    slope, intercept = p

    return slope, intercept

def plot_dissipation_scatter(ds, rolling_str='1d', color_by='TIME'):
    """
    Scatter plot of dissipation metrics vs epsilon terms, colored by month or another variable.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset with variables: EPSILON_TAU, epsilon_Q, DISSIPATION_LEM_TOTAL, MLD, TIME.
    rolling_str : str
        Resample frequency string (e.g., '1d', '6h').
    color_by : str
        Variable used to color the points. If 'TIME', colors by month.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure.
    axs : list of matplotlib.axes.Axes
        The list of subplot axes.
    """
    # Sort and resample
    ds = ds.sortby('TIME')
    if rolling_str:
        ds_rolling = ds.resample(TIME=rolling_str).mean(dim='TIME').sortby('TIME')
    else:
        ds_rolling = ds

    # Color setup
    if color_by == 'TIME':
        months = ds_rolling.TIME.dt.month
        cmap = plt.cm.twilight_shifted
        boundaries = np.arange(1, 14)
        norm = mcolors.BoundaryNorm(boundaries, cmap.N)
        color_vals = months
    else:
        color_vals = ds_rolling[color_by].values
        cmap = utilities.get_colormap(color_by)
        norm = None  # Let matplotlib handle it

    var1 = 'DISSIPATION_LEM_TOTAL'
    var2 = 'EPSILON_TAU'
    var3 = 'EPSILON_Q'

    with plt.style.context(plotting_style):
    # Set up figure
        fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

        # Scatter plots
        axs[0].scatter(ds_rolling[var2], ds_rolling[var1],
                    c=color_vals, cmap=cmap, norm=norm, s=5)

        axs[1].scatter(ds_rolling[var3], ds_rolling[var1],
                    c=color_vals, cmap=cmap, norm=norm, s=5)

        eps_sum = ds_rolling[var2] + ds_rolling[var3]
        sc = axs[2].scatter(eps_sum, ds_rolling[var1],
                            c=color_vals, cmap=cmap, norm=norm, s=5)
        
        ### add linear regression lines to the last subplot
        slope, intercept = fit_linear_regression(np.log10(eps_sum.values), np.log10(ds_rolling[var1].values))
        x_fit = np.logspace(np.log10(np.nanmin(eps_sum)), np.log10(np.nanmax(eps_sum)), 100)
        y_fit = 10**(slope * np.log10(x_fit) + intercept)
        axs[2].plot(x_fit, y_fit, color='red', linestyle='--', linewidth=1.5, label='Linear Fit')
        
        ### plot a 1:1 line for all three plots
        for ax in axs:
            ax.plot([1e-8, 1e-4], [1e-8, 1e-4], color='black', linestyle='--', linewidth=0.5, label='1:1 line')

        # Format axes
        for ax, xlabel in zip(
            axs,
            [utilities.get_label(var2) + f" [{utilities.get_unit(ds,var2)}]", utilities.get_label(var3) + f" [{utilities.get_unit(ds,var2)}]", utilities.get_label(var2) + ' + ' + utilities.get_label(var3)+ f" [{utilities.get_unit(ds,var2)}]"]
        ):
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlim(1e-8, 1e-4)
            ax.set_ylim(1e-8, 1e-4)
            ax.set_xlabel(xlabel)
            ax.grid(True)

        axs[0].set_ylabel(utilities.get_label(var1) + f"[{utilities.get_unit(ds,var1)}]")

        Glidemission = ds['GLIDER'][0].values + '/' + ds['MISSION'][0].values
        fig.suptitle(Glidemission)

        # Layout space for colorbar
        fig.tight_layout(rect=[0, 0, 0.95, 1])

        # Add colorbar
        cbar = fig.colorbar(sc, ax=axs, orientation='vertical', fraction=0.02, pad=0.02)

        if color_by == 'TIME':
            cbar.set_ticks(np.arange(1.5, 13.5, 1))
            cbar.ax.set_yticklabels([
                'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
            ])
            cbar.set_label("Month", fontsize=12)
        else:
            cbar.set_label(f"{color_by}", fontsize=12)

        plt.show()
    return fig, axs

def plot_histogram(ds, vars=['TEMP', 'PSAL'], bins: int = 50, log_scale: bool = False, style = "PDF", ax = None, **kwargs):
    """
    Plots histograms of the specified variables from the dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing the variables to plot.
    vars : list of str
        Variables to plot. Default is ['TEMP', 'PSAL'].
    bins : int
        Number of bins for the histogram. Default is 50.
    log_scale : list of bool
        If True, apply logarithmic scaling to the x-axis for the histogram.
        Default is False.
    density : bool
        If True, normalize histograms to show probability density.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the plot.
    axes : list of matplotlib.axes.Axes
        List of axis objects containing the plots.
    """
    if isinstance(vars, str):
        vars = [vars]

    if isinstance(log_scale, bool):
        log_scale = [log_scale] * len(vars)
    elif len(log_scale) != len(vars):
        raise ValueError("Length of log_scale must match length of vars.")

    num_vars = len(vars)
    n_cols = 3
    n_rows = int(np.ceil(num_vars / n_cols))

    with plt.style.context(plotting_style):
        if ax is None:  
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 6 * n_rows))
            force_plot = True
        else:
            fig = plt.gcf()
            axes = [ax]
            force_plot = False

        for i, var in enumerate(vars):
            if var not in ds:
                raise ValueError(f'Variable "{var}" not found in dataset.')

            data = ds[var].values.flatten()
            data = data[~np.isnan(data)]

            if log_scale[i]:
                data = data[data > 0]  # Remove non-positive values
                data = np.log10(data)
                x_label = f'log₁₀({utilities.get_label(var)} [{utilities.get_unit(ds, var)}])'
            else:
                x_label = f"{utilities.get_label(var)} ({utilities.get_unit(ds, var)})"

            ### take only the data above 0.5 % and below 99.5%
            min = np.nanpercentile(data,0.5)
            max = np.nanpercentile(data,99.5)

            axes = np.atleast_1d(axes).flatten()
            ax = axes[i]
            if style == "PDF":
                ax.hist(data, bins=bins, density=True, **kwargs)
                ax.set_ylabel('Probability Density')
            elif style == "Percentage":
                ax.hist(data, bins=bins, density=False, weights=np.ones_like(data) / len(data) * 100, **kwargs)
                ax.set_ylabel('Percentage (%)')
            else:
                print("Plotting the histogram as frequency counts. If you want to plot as PDF or Percentage, set style='PDF' or style='Percentage'")
                ax.hist(data, bins=bins, **kwargs)
                ax.set_ylabel('Frequency Counts')

            #if plot_MLE:
            #    ### calculate MLE and standard deviation from data
            #    MLE, sigma = stats.norm.fit(data, )  # returns (mu_hat, sigma_hat)
            #    label = f"MLE estimate: {MLE:.2e}"
            #   if log_scale == True:
            #        label = f"MLE estimate: {10**MLE:.2e}"
            #    ax.axvline(MLE, color='black', linestyle='--', label=label)

            ### plot vertical line for the median if log_scale and mean if not log_scale

            ax.set_xlabel(x_label)
            ax.set_title(f'Histogram of {utilities.get_label(var)}', fontsize=14)
            ax.set_xlim(min,max)

        # Hide unused subplots
        for j in range(num_vars, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        return fig, axes[:num_vars]


