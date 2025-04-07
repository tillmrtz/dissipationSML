import matplotlib.dates as mdates
import matplotlib.pyplot as plt
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
from dissipationSML import tools

import regionmask as rm
import ipywidgets as widgets
import numpy as np
import matplotlib.pyplot as plt
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
        max_depth = np.max(bath.elevation.values)  # Depths are negative
        max_level = level_spacing * (np.round(max_depth / level_spacing) + 1)
        levels = np.arange(0, max_level, level_spacing)
        contour_levels = levels[::2]  # Every second level
        return levels, contour_levels, max_level

def plot_glider_track(ds: xr.Dataset, ax: plt.Axes = None, **kw: dict) -> tuple({plt.Figure, plt.Axes}):
    """
    This function plots the glider track on a map, with latitude and longitude colored by time. Contour lines are added to the plot.

    Parameters
    ----------
    ds: xarray.Dataset
        Dataset with variables ** LATITUDE, LONGITUDE** and **TIME**
    ax: matplotlib.axes.Axes, default = None
        Existing Axes that the data should plotted to. 
    **kw: Optional; additional keyword arguments for the scatter plot.

    Returns
    -------
    fig: matplotlib.figure.Figure
        The figure object containing the plot.
    ax: matplotlib.axes.Axes
        The axis object containing the primary plot.
    
    """
    with plt.style.context(plotting_style):
        # Create figure and axis
        if ax is None:
            fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
        else:
            fig = plt.gcf()

        ## if dim is PROFILE_NUMBER, just take latitude, longitude and time values directly
        if 'DS_NUMBER' in ds.coords:
            latitudes, longitudes, times = zip(*[
                (
                    mission.LATITUDE.groupby(mission.PROFILE_NUMBER).mean(),
                    mission.LONGITUDE.groupby(mission.PROFILE_NUMBER).mean(),
                    mission.TIME.groupby(mission.PROFILE_NUMBER).mean()
                )
                for mission in (ds.sel(N_MEASUREMENTS=ds.DS_NUMBER == i, drop=True) for i in np.unique(ds.DS_NUMBER))
            ])

            latitudes = np.concatenate([lat.values for lat in latitudes])
            longitudes = np.concatenate([lon.values for lon in longitudes])
            times = np.concatenate([t.values for t in times])
        else:
            # Extract profile mean values
            latitudes = ds.LATITUDE.groupby(ds.PROFILE_NUMBER).mean().values
            longitudes = ds.LONGITUDE.groupby(ds.PROFILE_NUMBER).mean().values
            times = ds.TIME.groupby(ds.PROFILE_NUMBER).mean().values

        # Define map extent
        lon_lat_range = [
            np.min(longitudes) - 1, np.max(longitudes) + 1, 
            np.min(latitudes) - 1, np.max(latitudes) + 1
        ]
        ax.set_extent(lon_lat_range, crs=ccrs.PlateCarree())

        # Extract bathymetry data
        bath = bathymetry.sel(lon=slice(lon_lat_range[0], lon_lat_range[1]), 
                              lat=slice(lon_lat_range[2], lon_lat_range[3]))

        # Compute bathymetry levels
        levels, contour_levels, max_level = get_bathymetry_levels(bath)

        # Plot bathymetry as color mesh
        cmap = plt.get_cmap('Blues', len(levels))
        pcm = ax.pcolormesh(
            bath.lon, bath.lat, abs(bath.elevation.values), 
            cmap=cmap, vmin=0, vmax=max_level, transform=ccrs.PlateCarree())

        # Plot bathymetry contour lines
        ax.contour(
            bath.lon, bath.lat, abs(bath.elevation.values), 
            levels=contour_levels, colors='black', linewidths=0.5, transform=ccrs.PlateCarree())

        # Plot glider track (colored by time)
        sc = ax.scatter(longitudes, latitudes, c=times, cmap='inferno', s=10, **kw)

        # Colorbar for time (formatted date labels)
        cbar = plt.colorbar(sc, ax=ax)
        cbar.ax.set_yticklabels([pd.to_datetime(t).strftime('%Y-%b-%d') for t in cbar.get_ticks()])

        # Colorbar for bathymetry
        cbar_bath = plt.colorbar(pcm, ax=ax, label='Depth (m)', pad=0.02, shrink=0.8)
        cbar_bath.set_ticks(levels)

        # Add map features
        ax.add_feature(cfeature.LAND, color='lightgray', zorder=10)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8, zorder=11)

        # Labels and title
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title('Glider Track with Bathymetry')

        # Gridlines
        gl = ax.gridlines(draw_labels=True, color='black', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False

    return fig, ax


def plot_profile(ds: xr.Dataset, profile_num: int, use_raw: bool) -> tuple:
    """
    Plots temperature, salinity, and density against pressure on a single plot with three x-axes.

    Parameters
    ----------
    ds: xarray.Dataset
        Xarray dataset in OG1 format with at least PRESSURE, TEMPERATURE, SALINITY, and DENSITY.
    profile_num: int
        The profile number to plot.
    use_raw: bool
        If True, add the raw data to the plot.
    plot_binned: bool
        If True, add the binned data to the plot.
    binning: float
        The depth resolution for binning.

    Returns
    -------
    fig: matplotlib.figure.Figure
        The figure object containing the plot.
    ax: matplotlib.axes.Axes
        The axis object containing the primary plot.
    """
    with plt.style.context(plotting_style):  # Assuming `plotting_style` is defined elsewhere
        fig, ax1 = plt.subplots(figsize=(12, 9))  # Adjusted for profile visualization

        # Select the specific profile
        profile = ds.where(ds.PROFILE_NUMBER == profile_num, drop=True)
        pressures = profile.PRES.values
        depth = profile.DEPTH.values
        temperatures = profile.TEMP.values
        salinity = profile.PSAL.values
        density = profile.SIGMA_T.values
        #sigma0 = profile.SIGMA_T.mean().values

        mld = tools.calculate_mixed_layer_depth(density, depth)
        mld_CR = tools.calculate_MLD_with_CR(density, depth)

        mission = ds.id.split('_')[1][0:8]
        glider = ds.id.split('_')[0]

        # Temperature (Main X-Axis)
        ax1.plot(temperatures, depth, color='red', label='Temperature (°C)')
        ax1.set_xlabel('Temperature (°C)', color='red')
        ax1.tick_params(axis='x', colors='red', bottom=True, top=False, labelbottom=True, labeltop=False)

        # Salinity (Second X-Axis at Bottom)
        ax2 = ax1.twiny()
        ax2.plot(salinity, depth, color='blue', label='Salinity (PSU)')
        ax2.set_xlabel('Salinity (PSU)', color='blue')
        ax2.tick_params(axis='x', colors='blue', bottom=True, top=False, labelbottom=True, labeltop=False)
        ax2.xaxis.set_ticks_position('bottom')  # Move ticks to bottom
        ax2.spines['top'].set_visible(False)  # Hide top spine
        ax2.spines['bottom'].set_position(('axes', -0.07))  # Salinity (shift downward)

        # Density (Third X-Axis at Bottom, displayed as +1000)
        ax3 = ax1.twiny()
        ax3.plot(density, depth, color='grey', label='Density Anomaly (kg/m³)')
        ax3.set_xlabel('Density Anomaly (kg/m³)', color='grey')
        ax3.tick_params(axis='x', colors='grey', bottom=True, top=False, labelbottom=True, labeltop=False)
        ax3.xaxis.set_ticks_position('bottom')  # Move ticks to bottom
        ax3.spines['top'].set_visible(False)  # Hide top spine
        ax3.spines['bottom'].set_position(('axes', -0.14))

        if use_raw:
            salinity_raw = profile.PSAL_RAW.values
            temperature_raw = profile.TEMP_RAW.values
            density_raw = profile.SIGMA_T_RAW.values
            ### cut unrealistic values
            salinity_raw = np.where((salinity_raw > 35.5) | (salinity_raw < 34.8), np.nan, salinity_raw)
            density_raw = np.where((density_raw > 28.5) | (density_raw < 26.5), np.nan, density_raw)
            mld = tools.calculate_mixed_layer_depth(density_raw, depth)
            mld_CR = tools.calculate_MLD_with_CR(density_raw, depth)
            # Plot raw data
            ax1.plot(temperature_raw, depth, color='red', ls=':', label='Temperature Raw (°C)')
            ax2.plot(salinity_raw, depth, color='blue', ls=':', label='Salinity Raw (PSU)')
            ax3.plot(density_raw, depth, color='grey', ls=':', label='Density Anomaly Raw (kg/m³)')
            #ax1.axhline(y=mld_raw, color='black', linestyle=':', label=f'MLD Raw ({round(mld_raw,1)} m)')

        ax1.axhline(y=mld, color='black', linestyle='--', label=f'MLD ({round(mld,1)} m) from density threshold')
        ax1.axhline(y=mld_CR, color='black', linestyle=':', label=f'MLD ({round(mld_CR,1)} m) from CR')
        ### make a legend for all axes that are not on top of each other
        #[ax.legend(fontsize = 10) for ax in [ax1, ax2, ax3]]
        fig.legend(fontsize = 10)
        # Adjust x-axis label positions to avoid overlap
        ax1.xaxis.set_label_coords(0.5, -0.04)
        ax2.xaxis.set_label_coords(0.5, -0.13)
        ax3.xaxis.set_label_coords(0.5, -0.20)

        # Set pressure as y-axis (Increasing Downward)
        ax1.set_ylabel('Depth (m)')
        ax1.invert_yaxis()  # Pressure increases downward

        ax1.set_title(f'Profile {profile_num} ({glider}, mission: {mission})')

    return fig, ax1, ax2, ax3

def plot_profile_binned(ds: xr.Dataset, profile_num: int, binning: float,use_raw: bool,agg: str = 'mean') -> tuple:
    """
    Plots binned temperature, salinity, and density against depth on a single plot with three x-axes.

    Parameters
    ----------
    ds: xarray.Dataset
        Xarray dataset in OG1 format with at least PROFILE_NUMBER, DEPTH, TEMPERATURE, SALINITY, and DENSITY.
    profile_num: int
        The profile number to plot.
    binning: int
        The depth resolution for binning.
    use_raw: bool
        If True, use raw data for binning.
    agg: str
        The aggregation method to use for binning. Default is 'mean'. Other option is 'median'.

    Returns
    -------
    fig: matplotlib.figure.Figure
        The figure object containing the plot.
    ax1: matplotlib.axes.Axes
        The axis object containing the primary plot.
    """
    with plt.style.context(plotting_style):  # Assuming `plotting_style` is defined elsewhere
        fig, ax1 = plt.subplots(figsize=(12, 9))  # Adjusted for profile visualization

        # Select the specific profile
        profile = ds.where(ds.PROFILE_NUMBER == profile_num, drop=True)
        #sigma_0 = profile.SIGMA_T.mean().values
        if use_raw:
            vars = ['TEMP_RAW','PSAL_RAW','SIGMA_T_RAW']
        else:
            vars = ['TEMP','PSAL','SIGMA_T']

        binned_data = tools.bin_data(ds_profile = profile,vars=vars, resolution=binning, agg=agg)

        depth = binned_data['DEPTH']
        temperature = binned_data[vars[0]]
        salinity = binned_data[vars[1]]
        density = binned_data[vars[2]]

        ## cut off unrealistic values
        salinity = np.where((salinity > 35.5) | (salinity < 34.8), np.nan, salinity)
        density = np.where((density > 28.5) | (density < 26.5), np.nan, density)

        mld = tools.calculate_mixed_layer_depth(density, depth)
        mld_CR = tools.calculate_MLD_with_CR(density, depth)

        msk = np.isnan(depth)
        depth = depth[~msk]
        temperature = temperature[~msk]
        salinity = salinity[~msk]
        density = density[~msk]
        # Plot binned data
        mission = ds.id.split('_')[1][0:8]
        glider = ds.id.split('_')[0]
        min_depth = str(round(np.nanmin(profile.DEPTH.values),1))

        s=10+binning

        # Temperature (Main X-Axis)
        ax1.plot(temperature, depth , color='red', label='Temperature (°C)', ls='-')
        ax1.scatter(temperature, depth, color='red', marker='o',s=s)  # Scatter for visibility
        ax1.set_xlabel('Temperature (°C)', color='red')
        ax1.tick_params(axis='x', colors='red', bottom=True, top=False, labelbottom=True, labeltop=False)

        # Salinity (Second X-Axis at Bottom)
        ax2 = ax1.twiny()
        ax2.plot(salinity,depth, color='blue', label='Salinity (PSU)')
        ax2.scatter(salinity, depth, color='blue', marker='o',s=s)
        ax2.set_xlabel('Salinity (PSU)', color='blue')
        ax2.tick_params(axis='x', colors='blue', bottom=True, top=False, labelbottom=True, labeltop=False)
        ax2.xaxis.set_ticks_position('bottom')  # Move ticks to bottom
        ax2.spines['top'].set_visible(False)  # Hide top spine
        ax2.spines['bottom'].set_position(('axes', -0.07))  # Salinity (shift downward)

        # Density (Third X-Axis at Bottom, displayed as +1000)
        ax3 = ax1.twiny()
        ax3.plot(density, depth, color='grey', ls = '--', label='Density Anomaly (kg/m³)')
        ax3.scatter(density, depth, color='grey', marker='o',s=s)
        ax3.set_xlabel('Density Anomaly (kg/m³)', color='grey')
        ax3.tick_params(axis='x', colors='grey', bottom=True, top=False, labelbottom=True, labeltop=False)
        ax3.xaxis.set_ticks_position('bottom')  # Move ticks to bottom
        ax3.spines['top'].set_visible(False)  # Hide top spine
        ax3.spines['bottom'].set_position(('axes', -0.14))
        ### add a line for the mixed layer depth
        ax1.axhline(y=mld, color='black', linestyle='--', label=f'MLD ({round(mld,1)} m) from density threshold')
        ax1.axhline(y=mld_CR, color='black', linestyle=':', label=f'MLD ({round(mld_CR,1)} m) from CR')

        ### make a legend for all axes that are not on top of each other
        #[ax.legend(fontsize = 10) for ax in [ax1, ax2, ax3]]
        fig.legend(fontsize = 10)
        # Adjust x-axis label positions to avoid overlap
        ax1.xaxis.set_label_coords(0.5, -0.04)
        ax2.xaxis.set_label_coords(0.5, -0.13)
        ax3.xaxis.set_label_coords(0.5, -0.20)

        # Set pressure as y-axis (Increasing Downward)
        ax1.set_ylabel('Depth (m)')
        ax1.invert_yaxis()  # Pressure increases downward

        ax1.set_title(f'Profile {profile_num} ({glider}, mission: {mission})')

    return fig, ax1, ax2, ax3

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

        # Plot histogram of vertical distances
        ax.hist(distances, bins=20, color='blue', alpha=0.7)#,density=True)#,stacked=True)
        ax.set_xlabel('Vertical Distance (m)')
        ax.set_ylabel('Number of Measurements')
        ax.set_title(f'Profile {profile_num} Vertical Resolution')

    return fig, ax


def plot_min_max_depth(ds: xr.Dataset, bins= 20, ax = None, **kw: dict) -> tuple({plt.Figure, plt.Axes}):
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

def plot_MLD_evolution(ds,binning = None,use_raw = False, plot_density:bool = True) -> tuple:
    """
    This function plots the evolution of the mixed layer depth over time.

    Parameters
    ----------
    ds: xarray.Dataset
        Xarray dataset in OG1 format with at least TIME, DEPTH, TEMPERATURE, SALINITY, and DENSITY.
    binning: int
        The depth resolution for binning.
    use_raw: bool
        If True, use raw data for binning.

    Returns
    -------
    fig: matplotlib.figure.Figure
        The figure object containing the plot.
    ax: matplotlib.axes.Axes
        The axis object containing the primary plot.
    """
    time = []
    mld = []
    for i in np.unique(ds.PROFILE_NUMBER):
        profile = ds.where(ds.PROFILE_NUMBER == i, drop=True)
        time.append(profile.TIME.values[0])
        if use_raw:
            vars = ['SIGMA_T_RAW']
        else:
            vars = ['SIGMA_T']

        if binning:
            binned_data = tools.bin_data(ds_profile = profile,vars=vars, resolution=binning, agg='mean')
            depth = binned_data['DEPTH']
            density = binned_data[vars[0]]
        else:
            depth = profile.DEPTH.values
            density = profile.SIGMA_T.values

        mld.append(tools.calculate_mixed_layer_depth(density, depth))
        
    with plt.style.context(plotting_style):  # Assuming `plotting_style` is defined elsewhere
        fig, ax = plt.subplots(figsize=(18, 8), sharex=True)
        ax.plot(time, mld, color = 'black', label='MLD')                   
        ax.set_title('Mixed Layer Depth Evolution over Time')
        if plot_density:
            ### now plot the density profile
            density = ds.SIGTHETA_RAW.values
            depth = ds.DEPTH.values
            d = ax.scatter(ds.TIME.values,depth, c=density, s=20, cmap=cmo.dense,
                            vmin=np.nanpercentile(density, 0.5), vmax=np.nanpercentile(density, 99.5))
            fig.colorbar(d, ax=ax, label='Density [kg/m^3]')
            ax.set_ylim([np.nanmax(depth),0])
            ax.set_title(f'Mixed Layer Depth Evolution over Time with density profile (Binning: {binning} m)')
        else:
            ax.set_ylim([np.nanmax(mld)+10,0])
        ax.set_ylabel('Depth [m]')
        ax.set_xlabel('Time')
        # Dynamically adjust the interval based on the number of time points
        num_days = (time[-1] - time[0]).astype('timedelta64[D]').astype(int)
        interval = max(1, num_days // 25)  # Adjust the divisor to control the number of ticks
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=interval))
        #ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        #ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True)
        ax.legend()
        plt.tight_layout()
    
    return fig, ax

def plot_IFR_region_on_map(IFR_region: rm.Regions) -> tuple({plt.Figure, plt.Axes}):
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

def interactive_region_selector(default_coords):
    """
    Creates an interactive widget with four coordinate sliders (longitude & latitude)
    for defining a rectangular region. The region is only returned when the "Confirm Selection" button is clicked.

    Parameters
    ----------
    default_coords : list
        A list containing four pairs of default longitude and latitude values.

    Returns
    -------
    function
        A function `get_region()` that returns the selected `rm.Regions` object after confirmation.
    """
    
    def create_coord_sliders(default_long, default_lat):
        lat_slider = widgets.FloatSlider(description="Latitude", value=default_lat, min=58, max=67, orientation="vertical")
        long_slider = widgets.FloatSlider(description="Longitude", value=default_long, min=-16, max=-6)
        return widgets.VBox([lat_slider, long_slider])

    # Create a tab widget
    tab = widgets.Tab()
    coordinates = [create_coord_sliders(lon, lat) for lon, lat in default_coords]
    tab.children = coordinates

    # Set tab titles
    titles = ['First corner', 'Second corner', 'Third corner', 'Fourth corner']
    for i in range(4):
        tab.set_title(i, titles[i])

    # Output widget for the plot
    output = widgets.Output()

    # Region storage (updated only on confirmation)
    selected_region = {'region': None}

    # Function to update the plot
    def plot_rectangle():
        with output:
            clear_output(wait=True)  # Clear previous plot to prevent overlapping
            coords = np.array([[tab.children[i].children[1].value, 
                                tab.children[i].children[0].value] for i in range(4)])
            region = rm.Regions([coords], names=['IFR'])

            fig, ax = plot_IFR_region_on_map(region)
            display(fig)
            del fig, ax

    # Function to store the confirmed region
    def confirm_selection(b):
        coords = np.array([[tab.children[i].children[1].value, 
                            tab.children[i].children[0].value] for i in range(4)])
        selected_region['region'] = rm.Regions([coords], names=['IFR'])
        print("Region confirmed!")

    # Confirmation button
    confirm_button = widgets.Button(description="Confirm Selection")
    confirm_button.on_click(confirm_selection)

    # Attach event listeners to sliders for real-time plotting
    for i in range(4):
        for slider in tab.children[i].children:
            slider.observe(lambda change: plot_rectangle(), names='value')

    # Function to return the confirmed `rm.Regions` object
    def get_region():
        return selected_region['region']

    # Display widgets and plot
    display(tab, confirm_button, output)
    plot_rectangle()
    
    return get_region


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
"""
def plot_histograms(ds, vars: list, bins: int):

    This function plots histograms for the specified variables in a dataset.
    It also computes the sample mean, standard deviation (sigma), and variance (sigma^2),
    and adds them as vertical lines.

    Parameters
    ----------
    ds: xarray.Dataset
        Xarray dataset with the variables to plot.
    vars: list
        A list of variable names to plot.
    bins: int
        The number of bins for the histograms.

    Returns
    -------
    fig: matplotlib.figure.Figure
        The figure object containing the plot.
    axes: list
        List of axis objects containing the plots.
    
    num_vars = len(vars)
    cols = 2  # Number of columns per row
    rows = int(np.ceil(num_vars / cols))  # Determine number of rows needed

    fig, axes = plt.subplots(rows, cols, figsize=(12, 6 * rows))  # Adjust figure size dynamically
    axes = axes.flatten() if num_vars > 1 else [axes]  # Flatten in case of multiple subplots

    for i, var in enumerate(vars):
        ax = axes[i]
        data = ds[var].values.flatten()  # Convert to NumPy array for calculations
        
        mean_value = np.nanmean(data)  # Compute sample mean, ignoring NaNs
        std_dev = np.nanstd(data)  # Compute standard deviation (sigma), ignoring NaNs
        variance = std_dev ** 2  # Compute variance (sigma^2)

        ds[var].plot.hist(ax=ax, bins=bins, alpha=0.5, label=var)

        # Add vertical lines for mean and ± sigma (standard deviation)
        ax.axvline(mean_value, color='r', linestyle='dashed', linewidth=2, label=f'Mean μ: {mean_value:.2f}')
        ax.axvline(mean_value + std_dev, color='g', linestyle='dotted', linewidth=2, label=f'Std dev. σ: {std_dev:.2f}')
        ax.axvline(mean_value - std_dev, color='g', linestyle='dotted', linewidth=2)

        # Fetch description (long_name) from variable attributes
        var_desc = ds[var].attrs.get("long_name", var)
        unit = ds[var].attrs.get("units", "")

        # Update title to include mean, variance (σ²), and standard deviation (σ)
        ax.set_title(f"Histogram of {var_desc}\nμ={mean_value:.2f}, σ={std_dev:.2f}, σ²={variance:.2f}")

        ax.set_xlabel(f'{var} ({unit})')
        ax.set_ylabel('Frequency')
        ax.legend()

    # Remove any empty subplots if variables are not a multiple of cols
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    return fig, axes
"""

def plot_histograms(ds, vars: list, bins: int):
    """
    Plots histograms for the specified variables in a dataset.
    If "TIME" is included, it plots the difference between consecutive timestamps.

    Parameters
    ----------
    ds: xarray.Dataset
        Xarray dataset with the variables to plot.
    vars: list
        List of variable names to plot.
    bins: int
        The number of bins for the histograms.

    Returns
    -------
    fig: matplotlib.figure.Figure
        The figure object containing the plot.
    axes: list
        List of axis objects containing the plots.
    """
    num_vars = len(vars)
    cols = 2  # Number of columns per row
    rows = int(np.ceil(num_vars / cols))  # Number of rows needed

    fig, axes = plt.subplots(rows, cols, figsize=(12, 6 * rows))  # Dynamic figure size

    # Ensure axes is always iterable
    if num_vars == 1:
        axes = np.array([axes])  # Convert single axis to an array

    axes = axes.flatten()  # Flatten to ensure indexing works

    for i, var in enumerate(vars):
        ax = axes[i]

        # Handle "TIME" separately by computing differences
        if var == "TIME":
            time_values = ds[var].values.flatten()
            time_diffs = np.diff(time_values).astype('timedelta64[s]').astype(float)/3600  # Convert to seconds
            data = time_diffs
            xlabel = "Time Difference (hours)"
            var_desc = "Time Intervals"
            print('Max time difference:', np.max(time_diffs))
        else:
            data = ds[var].values.flatten()  # Convert to NumPy array for calculations
            xlabel = f"{var} ({ds[var].attrs.get('units', '')})"
            var_desc = ds[var].attrs.get("long_name", var)

        # Compute statistics
        mean_value = np.nanmean(data)  # Sample mean, ignoring NaNs
        std_dev = np.nanstd(data)  # Standard deviation (σ)
        variance = std_dev ** 2  # Variance (σ²)

        # Ensure `data` is not empty before plotting
        if len(data) > 0:
            ax.hist(data, bins=bins, alpha=0.5, label=var, color='steelblue', edgecolor='black')

            # Add vertical lines for mean and ± sigma
            ax.axvline(mean_value, color='r', linestyle='dashed', linewidth=2, label=f'Mean μ: {mean_value:.2f}')
            ax.axvline(mean_value + std_dev, color='g', linestyle='dotted', linewidth=2, label=f'Std dev. σ: {std_dev:.2f}')
            ax.axvline(mean_value - std_dev, color='g', linestyle='dotted', linewidth=2)

            # Set title and labels
            ax.set_title(f"Histogram of {var_desc}\nμ={mean_value:.2f}, σ={std_dev:.2f}, σ²={variance:.2f}")
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Frequency")
            ax.legend()
        else:
            ax.set_title(f"No data available for {var_desc}")

    # Remove empty subplots if variables are not a multiple of cols
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    return fig, axes


def plot_filtered_data(ds_filtered, vars, time_range=None):
    """
    Plots the original, filtered, and residual (difference) data for the given variables.
    The residual is plotted separately below the main plots.

    Parameters
    ----------
    ds_filtered : xarray.Dataset
        The dataset containing both the original and filtered variables.
    vars : list of str
        The variable names to plot (e.g., ['u10', 'v10']).
    time_range : tuple of str (optional)
        A tuple specifying the time range (start, end) in a format recognized by xarray (e.g., '2008-12-01', '2008-12-31').

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the plots.
    axes : list
        The list of axes objects.
    """
    num_vars = len(vars)

    # Apply time range if specified
    if time_range:
        ds_filtered = ds_filtered.sel(valid_time=slice(*time_range))

    # Create subplots: 2 rows per variable (original+filtered and residual)
    fig, axes = plt.subplots(num_vars * 2, 1, figsize=(20, 8 * num_vars), sharex=True)

    if num_vars == 1:
        axes = [axes]  # Ensure axes is always iterable

    for i, var in enumerate(vars):
        ax_main = axes[i * 2]  # Main plot (original + filtered)
        ax_residual = axes[i * 2 + 1]  # Residual plot

        residual = ds_filtered[var] - ds_filtered[f"{var}_hann_filtered"]

        # Plot original and filtered data
        ds_filtered[var].plot(ax=ax_main, label="Original", color="C0", linewidth=2)
        ds_filtered[f"{var}_hann_filtered"].plot(ax=ax_main, label="Filtered", color="C1", linewidth=2)

        ax_main.legend()
        ax_main.set_title(f"{var} (Original & Filtered)")
        ax_main.set_ylabel("m/s")

        # Plot residual separately
        residual.plot(ax=ax_residual, label="Residual", color="C2", linestyle="dashed")
        ax_residual.legend()
        ax_residual.set_title(f"{var} Residual (Original - Filtered)")
        ax_residual.set_ylabel("Residual (m/s)")

    axes[-1].set_xlabel("Time")  # Set xlabel only on the last subplot
    plt.tight_layout()
    plt.show()

    return fig, axes

def plot_winds_at_time(ds, time):
    """
    Plots wind vectors (u10, v10) at a specific time on a map.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing the wind variables ('u10' and 'v10').
    time : str or pandas.Timestamp
        The time point for which wind data should be plotted.
    
    Returns
    -------
    fig, ax : matplotlib figure and axis
        The wind quiver plot.
    """
    # Select data for the specified time
    ds_at_time = ds.sel(valid_time=time)

    # Extract wind components
    u = ds_at_time['u10'].values
    v = ds_at_time['v10'].values

    # Extract coordinates
    latitudes = ds['latitude'].values
    longitudes = ds['longitude'].values

    # Ensure data is 2D (reshape if necessary)
    if u.ndim == 1:
        lat_size = len(latitudes)
        lon_size = len(longitudes)
        u = u.reshape(lat_size, lon_size)
        v = v.reshape(lat_size, lon_size)

    # Create a meshgrid for quiver plotting
    lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)

    fig, ax = plt.subplots(figsize=(12, 6), subplot_kw={'projection': ccrs.PlateCarree()})

    # Define map extent
    ax.set_extent([longitudes.min()-1, longitudes.max()+1, latitudes.min()-1, latitudes.max()+1], crs=ccrs.PlateCarree())

    # Add map features
    ax.add_feature(cfeature.LAND, edgecolor='black')
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=":")

    # Downsample to avoid overcrowding
    step = 1  # Adjust for more/fewer arrows
    ax.quiver(
        lon_grid[::step, ::step], lat_grid[::step, ::step], 
        u[::step, ::step], v[::step, ::step], 
        transform=ccrs.PlateCarree(), scale=1000
    )

    ax.set_title(f"Wind Vectors at {time}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    # Gridlines
    gl = ax.gridlines(draw_labels=True, color='black', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False

    plt.show()
    return fig, ax
