import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib
from cmocean import cm as cmo
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from matplotlib.dates import DateFormatter
from scipy import stats
import matplotlib.colors as mcolors
import gsw
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import warnings
import os
from dissipationSML import tools

dir = os.path.dirname(os.path.realpath(__file__))
plotting_style = f"{dir}/plotting.mplstyle"
bathymetry = xr.open_dataset(f"{dir}/GEBCO_2024_IFR.nc")

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
        if ax is None:
            fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
        else:
            fig = plt.gcf()

        latitudes = ds.LATITUDE.values
        longitudes = ds.LONGITUDE.values
        times = ds.TIME.values

        # Plot latitude and longitude colored by time
        sc = ax.scatter(longitudes, latitudes, c=times, cmap='viridis',s=10, **kw)

        # Add colorbar with formatted time labels
        cbar = plt.colorbar(sc, ax=ax) #, label='Time')
        cbar.ax.set_yticklabels([pd.to_datetime(t).strftime('%Y-%b-%d') for t in cbar.get_ticks()])

        lon_lat_range = [np.min(longitudes)-1, np.max(longitudes)+1, np.min(latitudes)-1, np.max(latitudes)+1]

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
        

        ax.set_xlabel(f'Longitude')
        ax.set_ylabel(f'Latitude')
        ax.set_title('Glider Track')
        gl = ax.gridlines(draw_labels=True, color='black', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        #plt.show()

    return fig, ax


def plot_profile(ds: xr.Dataset, profile_num: int, plot_raw: bool) -> tuple:
    """
    Plots temperature, salinity, and density against pressure on a single plot with three x-axes.

    Parameters
    ----------
    ds: xarray.Dataset
        Xarray dataset in OG1 format with at least PRESSURE, TEMPERATURE, SALINITY, and DENSITY.
    profile_num: int
        The profile number to plot.
    plot_raw: bool
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

        mld = tools.calculate_mixed_layer_depth(density, depth)

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

        ax1.axhline(y=mld, color='black', linestyle='--', label=f'MLD ({round(mld,1)} m)')

        if plot_raw:
            salinity_raw = profile.PSAL_RAW.values
            temperature_raw = profile.TEMP_RAW.values
            density_raw = profile.SIGMA_T_RAW.values
            ### cut unrealistic values
            salinity_raw = np.where((salinity_raw > 35.5) | (salinity_raw < 34.8), np.nan, salinity_raw)
            density_raw = np.where((density_raw > 28.5) | (density_raw < 27), np.nan, density_raw)
            mld_raw = tools.calculate_mixed_layer_depth(density_raw, depth)
            # Plot raw data
            ax1.plot(temperature_raw, depth, color='red', ls=':', label='Temperature Raw (°C)')
            ax2.plot(salinity_raw, depth, color='blue', ls=':', label='Salinity Raw (PSU)')
            ax3.plot(density_raw, depth, color='grey', ls=':', label='Density Anomaly Raw (kg/m³)')
            ax1.axhline(y=mld_raw, color='black', linestyle=':', label=f'MLD Raw ({round(mld_raw,1)} m)')

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
        #plt.show()

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

        depth, temperature, salinity, density = tools.bin_data(ds_profile = profile,
                                                                resolution=binning , use_raw= use_raw, agg=agg)

        ## cut off unrealistic values
        salinity = np.where((salinity > 35.5) | (salinity < 34.8), np.nan, salinity)
        density = np.where((density > 28.5) | (density < 27), np.nan, density)

        mld = tools.calculate_mixed_layer_depth(density, depth)

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
        ax1.axhline(y=mld, color='black', linestyle='--', label=f'MLD ({round(mld,1)} m)')

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
        #plt.show()

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

        #plt.show()

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
        if force_plot:
            plt.show()
    return fig, ax

def plot_MLD_evolution(ds,binning = 1,use_raw = False, plot_density:bool = True) -> tuple:
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
        if binning >= 1:
            depth,_,_,density = tools.bin_data(profile, resolution=binning, use_raw=use_raw)
            mld.append(tools.calculate_mixed_layer_depth(density, depth))
        else:
            if use_raw:
                density = profile.SIGMA_T_RAW.values
                depth = profile.DEPTH.values
                mld.append(tools.calculate_mixed_layer_depth(density, depth))
            else:   
                density = profile.SIGMA_T.values
                depth = profile.DEPTH.values
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

