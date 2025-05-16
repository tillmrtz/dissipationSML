import yaml
import ipywidgets as widgets
from IPython.display import display, clear_output
import numpy as np
import regionmask as rm
from matplotlib import pyplot as plt
from cartopy import crs as ccrs
from cartopy import feature as cfeature
from dissipationSML.plotting import plot_IFR_region_on_map, plot_profile, plot_vertical_resolution, plot_dive_depth, plot_time_resolution, plot_CR


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

def interactive_profile(ds,profile_slider):
    """
    Creates an interactive profile viewer with external slider inputs.

    Parameters
    ----------
        ds : xarray.Dataset
            The dataset containing profile information.

    The function provides the following interactive widgets:
        - Profile Number: Selects the profile number from available profiles.
        - Variable 1, Variable 2, Variable 3: Dropdowns to choose up to three variables to plot.
        - Use Binned Data: Checkbox to toggle between raw and binned data.
        - Binning Resolution: Slider to adjust the binning resolution if binned data is used.

    Notes
    -----
    Original author: Till Moritz
    """

    def int_plot_profile(profile_num, var1, var2, var3, use_bins, binning):
        vars = [var1, var2, var3]
        fig, ax = plot_profile(ds, profile_num, vars, use_bins, binning)
        display(fig)
        plt.close(fig)
        del fig, ax

    # Variable selection dropdowns (with an empty option)
    # take only into account the variables with float or int data type
    var_options = [''] + [var for var in ds.data_vars if ds[var].dtype.kind in {'i', 'f'}]

    var1_dropdown = widgets.Dropdown(options=var_options, value=var_options[0], description="Var 1:")
    var2_dropdown = widgets.Dropdown(options=var_options, value=var_options[0], description="Var 2:")
    var3_dropdown = widgets.Dropdown(options=var_options, value=var_options[0], description="Var 3:")

    # Checkbox for using binned data
    use_bins_button = widgets.Checkbox(value=False, description='Bin Data')

    # Binning resolution slider
    binning_slider = widgets.FloatSlider(value=2, min=1, max=20, step=1,description='Res (m):',continuous_update=False)


    # Use a VBox to show/hide the binning slider based on the checkbox state
    binning_box = widgets.VBox([binning_slider])
    def toggle_binning_visibility(change):
        binning_box.layout.display = 'flex' if change['new'] else 'none'

    # Attach observer to toggle visibility
    use_bins_button.observe(toggle_binning_visibility, names='value')

    # Set initial visibility
    binning_box.layout.display = 'none' if not use_bins_button.value else 'flex'

    # Arrange variable dropdowns in a horizontal row
    var_selection_box = widgets.HBox([var1_dropdown, var2_dropdown, var3_dropdown])

    # Arrange all widgets in a vertical layout
    ui = widgets.VBox([widgets.Label("Select the profile number to visualize:"),profile_slider,
                       widgets.Label("Choose up to three variables to plot:"),var_selection_box,
                       widgets.Label("Additional settings:"),use_bins_button,binning_box])

    # Create interactive plot
    out = widgets.interactive_output(int_plot_profile, {
        'profile_num': profile_slider,
        'var1': var1_dropdown,
        'var2': var2_dropdown,
        'var3': var3_dropdown,
        'use_bins': use_bins_button,
        'binning': binning_slider
    })

    display(ui, out)
    
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
        plot_vertical_resolution(ds, profile_number)
        plot_time_resolution(ds, profile_number)

    
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

def interactive_MLD_profile(ds, profile_slider):
    """
    Creates an interactive Mixed Layer Depth (MLD) profile viewer where the method (CR or threshold) can be selected.
    Two plots are created for the selected profile: one with the density and the other with the cumulative convective resistance method.

    Parameters:
        ds : xarray.Dataset
            The dataset containing profile information.
        profile_slider : widgets.SelectionSlider
            Slider to select the profile number.
    """
    
    def plot_MLD_profile(profile_number):
        """Plots the selected MLD profile."""
        fig, ax = plt.subplots(1, 2, figsize=(18, 9))
        plot_profile(ds,profile_num=profile_number,vars = ['SIGMA_T'],ax1=ax[0],use_bins=True,binning=8)
        plot_CR(ds,profile_num=profile_number,use_bins=True,binning=8,ax=ax[1])
        display(fig)
        del fig, ax

    # Create interactive widget connection
    ui = widgets.interactive(plot_MLD_profile, profile_number=profile_slider)
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