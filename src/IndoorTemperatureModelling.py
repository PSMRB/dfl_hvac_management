"""
Script to predict one term of the indoor temperature using a FFNN.
Here is the equation
$\tau_{b, t+1}^{\rm in} = {\rm FFNN}_{\theta, b}(\tau_{t, b}^{\rm in}, \tau_{t, b}^{\rm amb}) + Z_{t, b}^{\rm TFT} & \forall t, b$
<=> Tin(t+1) = FFNN(Tin(t), Tamb(t)) + TFT(t)
"""

import itertools
from src.library.IndoorTemperatureModelling_Library import *
from src.library.Plot import correlation_coeff
import os
import pandas as pd
from torch.nn import ReLU, LeakyReLU

def main_indoor_temperature_modelling():
    """----------PATHS----------"""
    output_folderpath = os.path.abspath(
        "data/SmallOffice/Output-6zones_ASHRAE901_OfficeSmall_STD2022_Denver_1991-2020")
    idf_filepath = os.path.abspath("data/SmallOffice/6zones_ASHRAE901_OfficeSmall_STD2022_Denver.idf")
    epw_filepath = os.path.abspath("data/SmallOffice/USA_CO_Denver.Intl.AP.725650_US.Normals.1991-2020.epw")
    figure_folderpath = os.path.abspath(".data/SmallOffice/Figures/IndoorTemperatureModelling")
    training_data_filepath = os.path.join(output_folderpath, "Datasets/OutputVariablesTraining.h5")
    model_folderpath = os.path.join(output_folderpath, "../Models")
    c.createdir(figure_folderpath, up=1)
    c.createdir(model_folderpath, up=0)

    """----------PARAMETERS----------"""
    target_name = "Zone Mean Air Temperature(t+1)"  # target variable for the forecast
    seed = 34  # seed for random generation
    save_models = True  # whether to save the models or not
    save_figs = True  # whether to save the figures or not
    steps_btw_batches = 4 * 24  # number of steps between each batch to compute the 24h-forecast metrics
    zone_names = ['Attic', 'Core_ZN', 'Perimeter_ZN_1', 'Perimeter_ZN_2', 'Perimeter_ZN_3',
                 'Perimeter_ZN_4',
                 ]
    floors = [1] * 6
    zonesandfloors = zip(zone_names, floors)
    # HVAC system names (in the same order as the zone_names)
    ACUs = [f"PSZ-AC:{i}" for i in range(1, 6)]
    acusandfloors = zip(ACUs, [1] * 5)

    ### Subsequent variables/parameters
    # Create building object
    bldg = c.BuildingModel("6zones_ASHRAE901_OfficeSmall_STD2022_Denver", idf_filepath, epw_filepath,
                           output_folderpath, zonesandfloors, acusandfloors)

    # Flag the zones that are controlled
    for zone in bldg.zone_assets:
        zone_row = bldg.zones_df.loc[bldg.zones_df["name"] == zone.name].squeeze()
        zone.controlled = ~zone_row["is_plenum"]
        zone.acu = zone_row["ACU"]

    # zone_names whose temperature is actually controlled (not the plenum)
    controlled_zones = bldg.controlled_zone_names

    # name variables (useful only for the NN)
    input_names = [f"Zone Mean Air Temperature,{z}" for z in controlled_zones]  # indoor temperature
    input_names.extend([f"Air System Electricity Energy,{acu}[Wh]" for acu in ACUs])  # HVAC power
    input_names.extend(["Site Outdoor Air Drybulb Temperature,ENVIRONMENT"])  # outdoor temperature
    output_names1 = [f"Zone Mean Air Temperature(t+1),{z}" for z in controlled_zones]  # indoor temperature in t+1
    output_names2 = [f"Delta Mean Air Temperature(t+1),{z}" for z in controlled_zones]  # Delta indoor tempera in t+1

    """----------LOAD & CREATE DATASET----------"""
    loaded_df = pd.read_hdf(training_data_filepath, index=0, header=0)  # load data
    data = loaded_df

    # create the output columns
    shifted_data = pd.DataFrame(columns=output_names1 + output_names2, index=data.index)
    for z in controlled_zones:
        shifted_data[f"Zone Mean Air Temperature(t+1),{z}"] = data[f"Zone Mean Air Temperature,{z}"].shift(-1)
        shifted_data[f"Delta Mean Air Temperature(t+1),{z}"] = shifted_data[f"Zone Mean Air Temperature(t+1),{z}"] - \
                                                                      data[f"Zone Mean Air Temperature,{z}"]

    data = pd.concat([data, shifted_data], axis=1)  # add the output columns to the data



    data.dropna(inplace=True)  # remove the last row with NaN values

    """----------MODELS----------"""
    """RC model"""
    model_name = "RC"
    data_no_plenum = data.drop(data.filter(regex="Attic").columns, axis=1)  # remove the plenum zones from the dataset
    save_model_path = os.path.join(model_folderpath, model_name)
    save_figure_path = os.path.join(figure_folderpath, model_name)
    createdir(save_model_path)
    createdir(save_figure_path)
    print("\u0332".join("\nRC forecast:"))
    # perform all the NN operations (training, testing, saving, plotting) and return the normalized datasets and scalers
    delta_t = 1  # time step between each row of the dataset
    hp = {"nb_epochs_max": 500, "batch_size": 64, "nb_zones": len(controlled_zones), "lr": 1e-2,
          "patience_max": 20}
    for sd in [56710, 4567, 917]:  # 56710, 4567, 917
         rcmodelall(RCmodel, data_no_plenum, target_name, hp, save_model_path, save_models, seed=sd)

    """Neural Network"""
    model_name = "NN"
    output_names = output_names1 if target_name == "Zone Mean Air Temperature(t+1)" else output_names2
    data_nn = data_no_plenum.loc[:, input_names + output_names]  # load only the important variables
    # give a sign to the power input (positive if the HP is heating, negative if it is cooling)
    zonal_sensible_heating_rate = data_no_plenum.filter(axis=1, like="Zone Air System Sensible Heating Rate")
    zonal_sensible_cooling_rate = data_no_plenum.filter(axis=1, like="Zone Air System Sensible Cooling Rate")
    data_nn.loc[:, ["Air System Electricity Energy" in c for c in data_nn.columns]] = data_nn.filter(
        axis=1, like="Air System Electricity Energy").where(
        zonal_sensible_heating_rate.values > zonal_sensible_cooling_rate.values,
        -data_nn.filter(axis=1, like="Air System Electricity Energy").values
    ).div(1000)

    #                                       layers;      neurons; n° of repetitions
    for nl, nn, _ in itertools.product(range(1, 2, 1), [2, 5, 10], range(5), repeat=1):
        hyperparam = {
            "sparse": False,
            "activation": ReLU(),  # activation function (ReLU(), LeakyReLU())
            "nb_layers": nl,  # nb of hidden layers (does not count the input and output layer)
            "nb_neurons": nn,
            "batch_size": 128,
            "nb_epochs_max": 500,
            "patience_max": 20,  # number of epochs without improvement on the validation loss before stopping the training
            "regularization_coeff": 1e-3,
        }
        save_model_path = os.path.join(model_folderpath, model_name)
        save_figure_path = os.path.join(figure_folderpath, model_name)
        createdir(save_model_path)
        createdir(save_figure_path)
        print("\u0332".join("\nNeural Network forecast:"))
        # perform all the NN operations (training, testing, saving, plotting) and return the normalized datasets and scalers
        data_norm, db_norm, scalers = neuralnetworkall(data_nn, output_names, hyperparam, save_model_path,
                                                       save_figure_path,
                                                       save_models, save_figs, seed=None, seed_db=seed,
                                                       steps_btw_batches=steps_btw_batches)


if __name__ == "__main__":
    main_indoor_temperature_modelling()