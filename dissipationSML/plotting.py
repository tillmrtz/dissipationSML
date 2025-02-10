import matplotlib.dates as mdates
import matplotlib.pyplot as plt
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
from dissipationSML.tools import bin_data

dir = os.path.dirname(os.path.realpath(__file__))
plotting_style = f"{dir}/plotting.mplstyle"
bathymetry = xr.open_dataset(f"{dir}/GEBCO_2024_IFR.nc")

def plot_glider_track(ds: xr.Dataset, ax: plt.Axes = None, **kw: dict) -> tuple({plt.Figure, plt.Axes}):
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
        depths = profile.DEPTH.values
        temperatures = profile.TEMP.values
        salinity = profile.PSAL.values
        density = profile.SIGTHETA.values - 1000 # Display as deviations from 1000 kg/m³

        mission = ds.id.split('_')[1][0:8]
        glider = ds.id.split('_')[0]
        #min_pre = str(round(np.nanmin(pressures),1))

        # Temperature (Main X-Axis)
        ax1.plot(temperatures, pressures, color='red', label='Temperature (°C)')
        ax1.set_xlabel('Temperature (°C)', color='red')
        ax1.tick_params(axis='x', colors='red', bottom=True, top=False, labelbottom=True, labeltop=False)

        # Salinity (Second X-Axis at Bottom)
        ax2 = ax1.twiny()
        ax2.plot(salinity, pressures, color='blue', label='Salinity (PSU)')
        ax2.set_xlabel('Salinity (PSU)', color='blue')
        ax2.tick_params(axis='x', colors='blue', bottom=True, top=False, labelbottom=True, labeltop=False)
        ax2.xaxis.set_ticks_position('bottom')  # Move ticks to bottom
        ax2.spines['top'].set_visible(False)  # Hide top spine
        ax2.spines['bottom'].set_position(('axes', -0.07))  # Salinity (shift downward)

        if plot_raw:
            salinity_raw = profile.PSAL_RAW.values
            temperature_raw = profile.TEMP_RAW.values
            ### cut unrealistic values
            salinity_raw = np.where((salinity_raw > 36) | (salinity_raw < 34), np.nan, salinity_raw)
            # Plot raw data
            ax1.plot(temperature_raw, pressures, color='red', ls=':', label='Temperature Raw (°C)')
            ax2.plot(salinity_raw, pressures, color='blue', ls=':', label='Salinity Raw (PSU)')

        # Density (Third X-Axis at Bottom, displayed as +1000)
        ax3 = ax1.twiny()
        ax3.plot(density, pressures, color='grey', ls = '--', label='Density (+1000 kg/m³)')
        ax3.set_xlabel('Density (+1000 kg/m³)', color='grey')
        ax3.tick_params(axis='x', colors='grey', bottom=True, top=False, labelbottom=True, labeltop=False)
        ax3.xaxis.set_ticks_position('bottom')  # Move ticks to bottom
        ax3.spines['top'].set_visible(False)  # Hide top spine
        ax3.spines['bottom'].set_position(('axes', -0.14))
        ### make a legend for all axes that are not on top of each other
        #[ax.legend(fontsize = 10) for ax in [ax1, ax2, ax3]]
        fig.legend(fontsize = 10)
        # Adjust x-axis label positions to avoid overlap
        ax1.xaxis.set_label_coords(0.5, -0.04)
        ax2.xaxis.set_label_coords(0.5, -0.13)
        ax3.xaxis.set_label_coords(0.5, -0.20)

        # Set pressure as y-axis (Increasing Downward)
        ax1.set_ylabel('Pressure (dbar)')
        ax1.invert_yaxis()  # Pressure increases downward

        ax1.set_title(f'Profile {profile_num} ({glider}, mission: {mission})')#\nMin Pressure: {min_pre} dbar')
        #plt.show()

    return fig, ax1, ax2, ax3

def plot_profile_binned(ds: xr.Dataset, profile_num: int, binning: int,use_raw: bool) -> tuple:

    with plt.style.context(plotting_style):  # Assuming `plotting_style` is defined elsewhere
        fig, ax1 = plt.subplots(figsize=(12, 9))  # Adjusted for profile visualization

        # Select the specific profile
        profile = ds.where(ds.PROFILE_NUMBER == profile_num, drop=True)

        depth_binned, temperature_binned, salinity_binned, density_binned = bin_data(ds_profile = profile, resolution=binning , use_raw= use_raw)
        # Plot binned data
        mission = ds.id.split('_')[1][0:8]
        glider = ds.id.split('_')[0]
        min_depth = str(round(np.nanmin(profile.DEPTH.values),1))

        # Temperature (Main X-Axis)
        ax1.plot(temperature_binned, depth_binned, color='red', label='Temperature binned (°C)',ls = '-')
        ax1.set_xlabel('Temperature (°C)', color='red')
        ax1.tick_params(axis='x', colors='red', bottom=True, top=False, labelbottom=True, labeltop=False)

        # Salinity (Second X-Axis at Bottom)
        ax2 = ax1.twiny()
        ax2.plot(salinity_binned,depth_binned, color='blue', label='Salinity binned (PSU)')
        ax2.set_xlabel('Salinity (PSU)', color='blue')
        ax2.tick_params(axis='x', colors='blue', bottom=True, top=False, labelbottom=True, labeltop=False)
        ax2.xaxis.set_ticks_position('bottom')  # Move ticks to bottom
        ax2.spines['top'].set_visible(False)  # Hide top spine
        ax2.spines['bottom'].set_position(('axes', -0.07))  # Salinity (shift downward)

        # Density (Third X-Axis at Bottom, displayed as +1000)
        ax3 = ax1.twiny()
        ax3.plot(density_binned, depth_binned, color='grey', ls = '--', label='Density from binned data (+1000 kg/m³)')
        ax3.set_xlabel('Density (+1000 kg/m³)', color='grey')
        ax3.tick_params(axis='x', colors='grey', bottom=True, top=False, labelbottom=True, labeltop=False)
        ax3.xaxis.set_ticks_position('bottom')  # Move ticks to bottom
        ax3.spines['top'].set_visible(False)  # Hide top spine
        ax3.spines['bottom'].set_position(('axes', -0.14))
        ### make a legend for all axes that are not on top of each other
        #[ax.legend(fontsize = 10) for ax in [ax1, ax2, ax3]]
        fig.legend(fontsize = 10)
        # Adjust x-axis label positions to avoid overlap
        ax1.xaxis.set_label_coords(0.5, -0.04)
        ax2.xaxis.set_label_coords(0.5, -0.13)
        ax3.xaxis.set_label_coords(0.5, -0.20)

        # Set pressure as y-axis (Increasing Downward)
        ax1.set_ylabel('Pressure (dbar)')
        ax1.invert_yaxis()  # Pressure increases downward

        ax1.set_title(f'Profile {profile_num} ({glider}, mission: {mission})\n Min Depth: {min_depth} m)')
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
        depths = profile.PRES.values
        distances = np.abs(np.diff(depths))

        # Plot histogram of vertical distances
        ax.hist(distances, bins=20, color='blue', alpha=0.7,density=True)#,stacked=True)
        ax.set_xlabel('Vertical Distance (m)')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Profile {profile_num} Vertical Resolution')

        #plt.show()

    return fig, ax