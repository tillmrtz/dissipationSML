import yaml
import ipywidgets as widgets
from IPython.display import display, clear_output
import numpy as np
import regionmask as rm
from matplotlib import pyplot as plt
from cartopy import crs as ccrs
from cartopy import feature as cfeature
from dissipationSML.plotting import plot_IFR_region_on_map, plot_profile_binned, plot_profile, plot_vertical_resolution, plot_dive_depth


def load_glider_config(yaml_path):
    with open(yaml_path, 'r') as file:
        return yaml.safe_load(file)

def interactive_glider_selection(yaml_path):
    """
    Interactive function that displays all gliders and their dedicated missions. After confirming both statements,
    a directory with the url and the glider mission information is returned.

    Parameters:
    yaml_path (str): The path of the yaml file that summarizes the server url and each glider mission of interest

    Returns:
    dict: A dictionary that contains the exact server url to the glider's mission and it's information
    """
    config = load_glider_config(yaml_path)
    server_url = config['server_url']
    gliders = config['gliders']
    
    glider_names = [glider['name'] for glider in gliders]
    glider_dropdown = widgets.Dropdown(options=glider_names, description='Select Glider:')
    mission_dropdown = widgets.Dropdown(options=['Select a glider first'], description='Select Mission:', disabled=True)
    
    path_output = {'path': None,'dives': None,'glider':None,'mission':None}

    def update_missions(change):
        selected_glider = change['new']
        glider_info = next(glider for glider in gliders if glider['name'] == selected_glider)
        missions = [f"{m['date']} (dives: {m['dives']})" for m in glider_info['missions'] if m.get('folder') != 'no folder']
        
        if missions:
            mission_dropdown.options = missions
            mission_dropdown.disabled = False
        else:
            mission_dropdown.options = ['No available missions']
            mission_dropdown.disabled = True

    def confirm_selection(b):
        selected_glider = glider_dropdown.value
        selected_mission = mission_dropdown.value
        
        if selected_mission in ['No available missions', 'Select a glider first']:
            path_output['path'] = None
        else:
            mission_folder = next(
                mission['folder']
                for glider in gliders if glider['name'] == selected_glider
                for mission in glider['missions']
                if f"{mission['date']} (dives: {mission['dives']})" == selected_mission
            )
            path_output['path'] = f"{server_url}{mission_folder}/"
            path_output['dives'] = int(selected_mission.split('dives: ')[1].replace(')', ''))
            path_output['glider'] = selected_glider
            path_output['mission'] = mission_folder.split('/')[1]
        
        clear_output()
        display(glider_dropdown, mission_dropdown, confirm_button)
        print(f"Selected Path: {path_output['path']}")

    glider_dropdown.observe(update_missions, names='value')
    confirm_button = widgets.Button(description="Confirm Selection")
    confirm_button.on_click(confirm_selection)

    display(glider_dropdown, mission_dropdown, confirm_button)
    
    return path_output


def interactive_profile(ds, profile_slider, raw_button):
    """
    Creates an interactive profile viewer with external slider inputs.

    Parameters:
        ds : xarray.Dataset
            The dataset containing profile information.
        profile_slider : widgets.SelectionSlider
            Slider to select the profile number.
        raw_button : widgets.Checkbox
            Checkbox to toggle raw data.
    """
    def interactive_unbinned(profile_number, use_raw):
        """Plots the selected profile."""
        fig, ax1, ax2, ax3 = plot_profile(ds, profile_number, use_raw)
        display(fig)
        del fig, ax1, ax2, ax3

    # Create interactive widget connection
    ui = widgets.interactive(interactive_unbinned, profile_number=profile_slider, use_raw=raw_button)
    display(ui)

def interactive_profile_binned(ds, profile_slider, binning_slider, raw_button, agg_button):
    """
    Creates an interactive profile viewer with external slider inputs for the binned profile.

    Parameters:
        ds : xarray.Dataset
            The dataset containing profile information.
        profile_slider : widgets.SelectionSlider
            Slider to select the profile number.
        binning_slider : widgets.IntSlider
            Slider to select the binning value.
        raw_button : widgets.Checkbox
            Checkbox to toggle raw data.
        agg_button : widgets.ToggleButtons
            Toggle button to select aggregation method.
    """
    def interactive_binned(profile_number, binning, use_raw, agg):
        """Plots the selected profile."""
        fig, ax1, ax2, ax3 = plot_profile_binned(ds, profile_number, binning, use_raw, agg)
        display(fig)
        del fig, ax1, ax2, ax3

    # Create interactive widget connection
    ui = widgets.interactive(interactive_binned, profile_number=profile_slider, binning=binning_slider, use_raw=raw_button, agg=agg_button)
    display(ui)

def interactive_resolution_hist(ds, profile_slider):
    """
    Creates an interactive histogram viewer with external slider inputs for the vertical resolution.

    Parameters:
        ds : xarray.Dataset
            The dataset containing profile information.
        profile_slider : widgets.SelectionSlider
            Slider to select the profile number.
    """
    def interactive_res(profile_number):
        fig, ax = plot_vertical_resolution(ds, profile_number)
        display(fig)
        del fig, ax
    
    # Create interactive widget connection
    ui = widgets.interactive(interactive_res, profile_number=profile_slider)
    display(ui)

def interactive_dive_profile(ds, dive_slider):
    """
    Creates an interactive depth profile viewer using ipywidgets.

    Parameters:
        ds : xarray.Dataset
            The dataset containing profile information.
        dive_slider : widgets.SelectionSlider
            Slider to select the dive number.
        raw_button : widgets.Checkbox
            Checkbox to toggle raw data.
    """
    def plot_dive_profile(dive_number):
        """Plots the selected depth profile."""
        fig, ax = plot_dive_depth(ds, dive_number)
        display(fig)
        del fig, ax
    
    # Create interactive widget connection
    ui = widgets.interactive(plot_dive_profile, dive_number=dive_slider)
    display(ui)



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

def interactive_bad_profile_checking(ds):
    """
    Creates an interactive profile viewer using ipywidgets.

    Parameters:
        ds : xarray.Dataset
            The dataset containing profile information.
    """
    bad_profiles = []

    def plot_profile(profile_number, binning, use_raw, agg):
        """Plots a selected profile."""
        fig, ax1, ax2, ax3 = plot_profile_binned(ds, profile_number, binning, use_raw, agg)
        display(fig)
        del fig, ax1, ax2, ax3

    def mark_bad_profile(profile_number):
        """Marks a profile as bad."""
        if profile_number not in bad_profiles:
            bad_profiles.append(int(profile_number))
        print(f"Marked profile {profile_number} as bad. Current bad profiles: {bad_profiles}")

    def reset_bad_profiles():
        """Resets the bad profiles list."""
        nonlocal bad_profiles
        bad_profiles = []
        print("Bad profiles list reset.")

    # Create widgets
    profile_slider = widgets.SelectionSlider(
        options=np.unique(ds.PROFILE_NUMBER.values).astype(int), 
        description='Profile number:', 
        continuous_update=False
    )
    binning_slider = widgets.IntSlider(value=2, min=1, max=20, description='Binning (m):')
    raw_button = widgets.Checkbox(value=False, description='Use raw data')
    agg_button = widgets.ToggleButtons(options=['mean', 'median'], description='Aggregation:')
    bad_button = widgets.Button(description="Mark as Bad")
    reset_button = widgets.Button(description="Reset Bad Profiles")
    output = widgets.Output()
    
    display(widgets.VBox([
        profile_slider, binning_slider, raw_button, agg_button, bad_button, reset_button, output,
        widgets.interactive_output(plot_profile, {'profile_number': profile_slider, 'binning': binning_slider, 'use_raw': raw_button, 'agg': agg_button})
    ]))
    
    def on_bad_button_clicked(b):
        """Handles marking a profile as bad."""
        with output:
            clear_output(wait=True)
            mark_bad_profile(profile_slider.value)
    
    def on_reset_button_clicked(b):
        """Handles resetting bad profiles."""
        with output:
            clear_output(wait=True)
            reset_bad_profiles()
    
    bad_button.on_click(on_bad_button_clicked)
    reset_button.on_click(on_reset_button_clicked)
    
    return bad_profiles  # Returns bad profiles when done
