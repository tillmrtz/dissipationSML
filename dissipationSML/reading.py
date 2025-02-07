import xarray as xr
import os
from bs4 import BeautifulSoup
import requests
from tqdm import tqdm
import os
import requests
from seagliderOG1 import convertOG1, writers

def download_file_from_server(file_name, server_url, destination_path):
    """
    Download a file from the server and save it to the appropriate directory.
    
    The folder structure will be:
    destination_path/
        └── glider_number/
            └── mission_date/
                └── file_name.nc

    Parameters:
    file_name (str): The name of the file to download (e.g., 'p0050001_20080606.nc').
    server_url (str): The base URL of the server where the file is hosted.
    destination_path (str): The base directory where files should be saved.

    Returns:
    str: The path to the downloaded file.
    """
    # Extract glider number and mission date from the filename
    glider_number = server_url.split('uw/')[-1].split('/')[0]
    mission_date = server_url.split('/')[-2]
    
    # Construct directory paths
    glider_folder = os.path.join(destination_path, glider_number)
    mission_folder = os.path.join(glider_folder, mission_date)
    
    # Create directories if they don't exist
    os.makedirs(mission_folder, exist_ok=True)
    
    # Full path to the destination file
    destination_file = os.path.join(mission_folder, file_name)
    
    # Check if file already exists
    if os.path.exists(destination_file):
        print(f"File already exists: {destination_file}")
        return destination_file

    # Download the file if it doesn't exist
    response = requests.get(f"{server_url}/{file_name}", stream=True)
    
    if response.status_code == 200:
        with open(destination_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded: {file_name} to {destination_file}")
    else:
        raise Exception(f"Failed to download {file_name}, status code: {response.status_code}")
    
    return destination_file



def load_sample_dataset(dataset_name, server_url, destination_path="."):
    """
    Download and load a sample dataset from the server.

    Parameters:
    dataset_name (str): Name of the dataset file (e.g., 'p0150500_20050213.nc').
    server_url (str): Base URL of the server where the file is hosted.
    destination_path (str): Directory where the file will be saved.

    Returns:
    xarray.Dataset: The loaded dataset.
    """
    file_path = download_file_from_server(dataset_name, server_url, destination_path)
    return xr.open_dataset(file_path, decode_timedelta=True)


def list_files_in_https_server(url):
    """
    List NetCDF files in an HTTPS server directory.

    Parameters:
    url (str): The URL to the directory containing the files.

    Returns:
    list: A list of filenames found in the directory.
    """
    response = requests.get(url)
    response.raise_for_status()  # Raise an error for bad status codes

    soup = BeautifulSoup(response.text, "html.parser")
    files = [link.get("href") for link in soup.find_all("a") if link.get("href").endswith(".nc")]

    return files


def filter_files_by_profile(file_list, start_profile=None, end_profile=None):
    """
    Filter files based on the start and end profile numbers.

    Parameters:
    file_list (list): List of filenames to filter.
    start_profile (int, optional): The starting profile number to filter files.
    end_profile (int, optional): The ending profile number to filter files.

    Returns:
    list: A list of filtered filenames.
    """
    filtered_files = []
    for file in file_list:
        if file.endswith(".nc"):
            profile_number = int(file[5:8])  # Extract the profile number from the filename
            if ((start_profile is None or profile_number >= start_profile) and
                (end_profile is None or profile_number <= end_profile)):
                filtered_files.append(file)
    return filtered_files


def read_basestation(server_url, destination_path=".", start_profile=None, end_profile=None):
    """
    Download and load datasets from a server, optionally filtering by profile range.

    Parameters:
    server_url (str): The URL of the server containing the NetCDF files.
    destination_path (str): Path to save the downloaded files.
    start_profile (int, optional): The starting profile number to filter files.
    end_profile (int, optional): The ending profile number to filter files.

    Returns:
    list: A list of xarray.Dataset objects loaded from the filtered NetCDF files.
    """
    file_list = list_files_in_https_server(server_url)
    filtered_files = filter_files_by_profile(file_list, start_profile, end_profile)
    
    datasets = []
    for file in filtered_files:
        ds = load_sample_dataset(file, server_url=server_url, destination_path=destination_path)
        datasets.append(ds)
    
    return datasets



import yaml
import ipywidgets as widgets
from IPython.display import display, clear_output

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


def convert_with_variables(datasets, variables_needed):
    """
    Converts input datasets to the OG1 format, filters for the specified variables,  
    and renames the 'divenum' variable to 'DIVE_NUMBER'.  

    Args:  
        datasets (list or iterable): A collection of datasets to be converted.  
        variables_needed (list): A list of variable names to retain in the converted dataset.  

    Returns:  
        xarray.Dataset: The transformed dataset with selected variables and renamed column.  
    """
    # Check first if variables are in the dataset, if not, raise an error
    ds = convertOG1.convert_to_OG1(datasets)
    ds = ds.rename_vars({'divenum': 'DIVE_NUMBER'})
    for var in variables_needed:
        if var not in ds.variables:
            raise ValueError(f"Variable '{var}' not found in dataset. Possible variables are: {list(ds.variables)}")
    ds = ds[variables_needed]
    
    return ds

