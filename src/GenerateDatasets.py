"""
This file:
    1) Generates a .csv from eplusout.eso (the output of EnergyPlus without the meters) and saves it in run folder
    2) Save the same data and the meters under the .xlsx and hdf formats
                                                            (file named OutputVariablesEnergyPlus in Datasets folder)
    3) Plot the data to analyze the dataset and save figures in Figures/YearlyAnalysis.
    4) Create the linear regression for both COP and saves their parameters in a pickle file in Models/COP. 3D figures
        are also saved in Figures/COP.
    5) Overwrite the electricity consumption of the HP in OutputVariablesEnergyPlus with the one computed by
        the COP models to form OutputVariablesTraining saved in Datasets folder.
    6) Analyse the error made during this overwriting and plot the distribution of the error. It is saved in
        Figures/COP folder.
"""

import os
from src.library.GenerateDatasets_Library import generate_datasets, get_variable_list
from src.library import Common as c

# call the function on the model using the IWEC weather file
def main_generate_datasets():
    """
    Generate the datasets for the 6 zones small office model in Denver, CO, USA.
    Two time periods are generated: 1991-2020 and 2006-2020.
    """
    # generate two years of data for the model
    timeperiods = ["1991-2020", "2006-2020"]
    for tp in timeperiods:
        # paths to (1) the output directory; (2) the file of the EnergyPlus model; (3) the weather file
        output_folderpath = os.path.abspath(f"data/SmallOffice/Output-6zones_ASHRAE901_OfficeSmall_STD2022_Denver_{tp}")
        idf_filepath = os.path.abspath("data/SmallOffice/6zones_ASHRAE901_OfficeSmall_STD2022_Denver.idf")
        epw_filepath = os.path.abspath(f"data/SmallOffice/USA_CO_Denver.Intl.AP.725650_US.Normals.{tp}.epw")

        save_fig = True
        frequency = 4  # "TimeStep" or "Hourly"
        polynom_degree = 3  # polynom degree for approximating the COP relationship
        zones = ['Attic', 'Core_ZN', 'Perimeter_ZN_1', 'Perimeter_ZN_2', 'Perimeter_ZN_3',
                 'Perimeter_ZN_4',
                 ]
        floors = [1] * 6
        zonesandfloors = zip(zones, floors)
        # HVAC system names (in the same order as the zone_names)
        ACUs = [f"PSZ-AC:{i}" for i in range(1, 6)]
        acusandfloors = zip(ACUs, [1] * 5)

        # Create building object
        bldg = c.BuildingModel((os.path.basename(idf_filepath)).split(".")[0], idf_filepath, epw_filepath,
                               output_folderpath, zonesandfloors, acusandfloors)

        variables = get_variable_list(bldg)

        bldg.set_simulationvariables(variables, simulationfrequency=4, desiredfrequency=1)

        generate_datasets(bldg, save_fig)

