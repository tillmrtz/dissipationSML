The notebooks presented here are include the whole data analysis. The order in which they were performed is explained in the following:

1. **convert_to_OG1.ipynb**:
- Loads the seaglider data and converts it to OG1 format
- If not yet downloaded, a folder is created for each glider mission
- Vertical velocities are calculated from modelled and measured vertical velocity of the glider
- Datset is saved in the same folder as each profile data

2. **dissipation_one_glider.ipynb**:
- Load seaglider data for one mission
- Calculate dissipation using LEM:
    - 
- Compare to sg005 mission for calculation of proportionality constant