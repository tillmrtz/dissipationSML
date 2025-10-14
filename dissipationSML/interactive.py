import yaml
import ipywidgets as widgets
from IPython.display import display, clear_output
import numpy as np
import regionmask as rm
from matplotlib import pyplot as plt
from cartopy import crs as ccrs
from cartopy import feature as cfeature
from dissipationSML import tools
from dissipationSML.plotting import plot_IFR_region_on_map, plot_profile, plot_vertical_resolution, plot_dive_depth, plot_time_resolution, plot_CR
import os
dir = os.path.dirname(os.path.realpath(__file__))
plotting_style = f"{dir}/plotting.mplstyle"


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

def interactive_profile(ds, mld_df=None):
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

    def plot_func(profile_num, var1, var2, var3, use_bins, binning):
        vars = [var1, var2, var3]
        fig, ax = plot_profile(ds, profile_num, vars, use_bins, binning)
        if mld_df is not None:
            MLD = mld_df[mld_df['PROFILE_NUMBER'] == profile_num]['MLD'].values[0]
            ### plot MLD line
            if not np.isnan(MLD):
                ax.axhline(MLD, color='black', linestyle='--', label=f'MLD ({MLD:.2f} m)')
                ax.legend()
        display(fig)
        plt.close(fig)
        del fig, ax

    profile_slider = widgets.SelectionSlider(options=np.unique(ds.PROFILE_NUMBER.values).astype(int), description='Profile:', continuous_update=False)

    # Variable selection dropdowns (with an empty option)
    # take only into account the variables with float or int data type
    var_options = ['','TIME'] + [var for var in ds.data_vars if ds[var].dtype.kind in {'i', 'f'}]
    ### also add the possible coordinates
    var_options += [var for var in ds.coords if ds[var].dtype.kind in {'i', 'f'}]

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
    out = widgets.interactive_output(plot_func, {
        'profile_num': profile_slider,
        'var1': var1_dropdown,
        'var2': var2_dropdown,
        'var3': var3_dropdown,
        'use_bins': use_bins_button,
        'binning': binning_slider
    })

    display(ui, out)
    
def interactive_resolution_hist(ds):
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

    profile_slider = widgets.SelectionSlider(options=np.unique(ds.PROFILE_NUMBER.values).astype(int), description='Profile:', continuous_update=False)    
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

threshold_vars = {
    "DENSITY": "In situ density",
    "SIGMA": "Sigma-t",
    "SIGMA_T": "Sigma-t",
    "SIGMA_1": "Sigma-1",
    "POTDENS0": "Potential density (σ₀)",
    "SIGTHETA": "Potential density (σθ)",
    "PSAL": "Practical salinity",
    "SA": "Absolute salinity",
    "TEMP": "Temperature",
    "THETA": "Potential temperature"}

def interactive_mld_profile(ds):
    """
    Interactive profile viewer with MLD calculations (threshold & CR) and optional binning.
    """

    def plot_func_mld(profile_num, var, use_threshold_method, ref_depth, threshold,
                     use_CR_method, CR_threshold, use_bins, binning):
        profile = ds.where(ds.PROFILE_NUMBER == profile_num, drop=True)
        vars_to_plot = [var] if var else []
        mld_threshold, mld_CR = None, None

        # --- Compute Threshold MLD ---
        if use_threshold_method:
            mld_threshold = tools.mld_profile_treshhold(
                profile=profile,
                variable=var,
                ref_depth=ref_depth,
                threshold=threshold,
                use_bins=use_bins,
                binning=binning
            )

        # --- Compute CR MLD ---
        has_sigma1 = 'SIGMA_1' in ds.data_vars
        if use_CR_method:
            if not has_sigma1:
                # Plot dummy with warning box
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.text(0.5, 0.5,
                        "SIGMA_1 (density anomaly) is required for CR method.\n\n"
                        "Please run:\n  tools.add_sigma_1(ds)\nto add it to the dataset.",
                        fontsize=14, color='red', ha='center', va='center', wrap=True)
                ax.axis('off')
                display(fig)
                plt.close(fig)
                return
            else:
                mld_CR = tools.mld_profile_CR(
                    profile=profile,
                    threshold=CR_threshold,
                    use_bins=use_bins,
                    binning=binning
                )
                if 'SIGMA_1' not in vars_to_plot:
                    vars_to_plot.append('SIGMA_1')
                with plt.style.context(plotting_style):
                    fig, ax = plt.subplots(1, 2, figsize=(22, 10), sharey=True)
                    plot_profile(ds, profile_num, vars_to_plot, use_bins, binning, ax=ax[0])
                    plot_CR(ds, profile_num, ax=ax[1])
                    ax1 = ax[0]
                    ax1.invert_yaxis()

        else:
            fig, ax1 = plot_profile(ds, profile_num, vars_to_plot, use_bins, binning)

        # --- Add MLD lines to plot ---
        if mld_threshold is not None:
            ax1.axhline(mld_threshold, color='red', linestyle='--', label=f'MLD Threshold ({mld_threshold:.2f} m)')

        if mld_CR is not None:
            ax1.axhline(mld_CR, color='blue', linestyle='--', label=f'MLD CR ({mld_CR:.2f} m)')

        ax1.legend(loc='upper right')
        plt.show()

    # --- Widgets ---
    profile_slider = widgets.SelectionSlider(options=np.unique(ds.PROFILE_NUMBER.values).astype(int), description='Profile:', continuous_update=False)

    # take as var options only the variables from threshold_vars that are present in the dataset
    var_options = [var for var in threshold_vars.keys() if var in ds.data_vars]
    var_dropdown = widgets.Dropdown(options=var_options, value=var_options[0], description="Plot variable:")

    # Threshold method widgets
    calc_MLD_threshold = widgets.Checkbox(value=False, description='Use Threshold Method')
    threshold_slider = widgets.FloatSlider(value=0.03, min=0.005, max=0.1, step=0.005,
                                           description='Threshold (Δ kg/m³):', continuous_update=False)
    ref_depth_slider = widgets.FloatSlider(value=5, min=0, max=20, step=1,
                                           description='Ref Depth (m):', continuous_update=False)
    threshold_box = widgets.VBox([ref_depth_slider, threshold_slider])
    threshold_box.layout.display = 'none'

    def toggle_threshold_visibility(change):
        threshold_box.layout.display = 'flex' if change['new'] else 'none'
    calc_MLD_threshold.observe(toggle_threshold_visibility, names='value')

    # CR method widgets
    calc_MLD_CR = widgets.Checkbox(value=False, description='Use CR Method (uses SIGMA_1)')
    CR_threshold_slider = widgets.FloatSlider(value=-1, min=-10, max=-0.5, step=0.5,
                                              description='CR Threshold:', continuous_update=False)
    CR_box = widgets.VBox([CR_threshold_slider])
    CR_box.layout.display = 'none'

    def toggle_CR_visibility(change):
        CR_box.layout.display = 'flex' if change['new'] else 'none'
    calc_MLD_CR.observe(toggle_CR_visibility, names='value')

    # Binning widgets
    use_bins_button = widgets.Checkbox(value=False, description='Bin Data')
    binning_slider = widgets.FloatSlider(value=2, min=1, max=20, step=1,
                                         description='Resolution (m):', continuous_update=False)
    binning_box = widgets.VBox([binning_slider])
    binning_box.layout.display = 'none'

    def toggle_binning_visibility(change):
        binning_box.layout.display = 'flex' if change['new'] else 'none'
    use_bins_button.observe(toggle_binning_visibility, names='value')

    # --- Layout ---
    ui = widgets.VBox([
        widgets.Label("Select the profile number to visualize:"),
        profile_slider,
        widgets.Label("Choose the variable for threshold method:"),
        var_dropdown,
        calc_MLD_threshold,
        threshold_box,
        calc_MLD_CR,
        CR_box,
        widgets.Label("Additional settings:"),
        use_bins_button,
        binning_box
    ])

    # --- Interactive Output ---
    out = widgets.interactive_output(plot_func_mld, {
        'profile_num': profile_slider,
        'var': var_dropdown,
        'use_threshold_method': calc_MLD_threshold,
        'ref_depth': ref_depth_slider,
        'threshold': threshold_slider,
        'use_CR_method': calc_MLD_CR,
        'CR_threshold': CR_threshold_slider,
        'use_bins': use_bins_button,
        'binning': binning_slider
    })

    display(ui, out)

