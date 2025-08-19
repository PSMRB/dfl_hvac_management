"""
Main file to run the deterministic optimization problem
"""
from itertools import product
import numpy as np
import library.Common as c
from library.EnergyManagementSystem import EMS
from library.EnergyPlusAPI import EnergyPlusSimulator
from library.Building import BuildingModel
from library.myomlt.neuralnet.nn_formulation import (
    ReluBigMFormulation, ReluIdealFormulation, ReluExtendedFormulation1, ReluExtendedFormulation2)
from library.OptimizationProblem_Deterministic_Lib import save_results, modify_idf
from library.GenerateDatasets_Library import get_variable_list
import pandas as pd
import os

####################################
### Parameters
####################################
# N.B., the file 18zones_ASHRAE901_OfficeMedium_STD2019_Denver.idf does not account for daylight saving time.
# Also, it is recommended to perform all the operation in UTC time and convert the results to the local time zone
# only for visualization purposes.

# Read the medoids days
medoids_dates = pd.read_csv("optimization_parameters/ClusteredDays.csv", index_col=0, parse_dates=True)
figure_folderpath = os.path.abspath("../BuildingModel/Figures/Optimization")
c.createdir(figure_folderpath, up=1)
thermodynamics_models = ["nn_bigm"]  # only one formulation implemented in cvxpy layer model for now
nb_epochs = 1

for ep in range(nb_epochs):
    for medoid_name, thermodynamics_model in product(medoids_dates.index, thermodynamics_models):
        T0_date = medoids_dates.loc[medoid_name, "Date"]
        T0 = pd.Timestamp(T0_date, tz="UTC")  # starting datetime of the simulation
        # T0 = pd.Timestamp("01/01/2017", tz="UTC")
        T_ems = 24  # number of hours in the EMS
        dt_ems = 1  # time step period of the EMS in hour (e.g. 0.25 for 15min)
        T_sim = 24  # number of hours in the simulation
        dt_sim = 0.25  # time step period of the simulation in hour (e.g. 0.25 for 15min)
        ems = EMS()
        ems.set_times(T0, T_ems, dt_ems, T_sim, dt_sim)
        nn_hyperparameters = {"nb_layers": 3, "nb_neurons": 10}

        ####################################
        ### Create the BuildingModel object
        ####################################

        # paths
        output_folderpath = os.path.abspath("../BuildingModel/Output-18zones_ASHRAE901_OfficeMedium_STD2019_Denver")
        idf_filepath = os.path.abspath("../BuildingModel/18zones_ASHRAE901_OfficeMedium_STD2019_Denver.idf")
        epw_filepath = os.path.abspath("../BuildingModel/USA_CO_Denver.Intl.AP.725650_US.Normals.2006-2020.epw")
        training_data_filepath = os.path.join(output_folderpath, "../Datasets/OutputVariablesTraining.h5")
        model_folderpath = os.path.join(output_folderpath, "../Models")
        results_summary_filepath = os.path.normpath("../OptimizationResults/ResultsSummary.xlsx")
        c.createdir(os.path.dirname(results_summary_filepath), up=1)

        # parameters
        zone_names = ['FirstFloor_Plenum',
                      'Core_bottom', 'Perimeter_bot_ZN_1', 'Perimeter_bot_ZN_2', 'Perimeter_bot_ZN_3',
                      'Perimeter_bot_ZN_4',
                      'MidFloor_Plenum', 'Core_mid', 'Perimeter_mid_ZN_1', 'Perimeter_mid_ZN_2', 'Perimeter_mid_ZN_3',
                      'Perimeter_mid_ZN_4',
                      'TopFloor_Plenum', 'Core_top', 'Perimeter_top_ZN_1', 'Perimeter_top_ZN_2', 'Perimeter_top_ZN_3',
                      'Perimeter_top_ZN_4'
                      ]
        floors = [1] * 6 + [2] * 6 + [3] * 6
        zonesandfloors = zip(zone_names, floors)
        # HVAC system names (in the same order as the zone_names)
        ACUs = ["PACU_VAV_BOT", "PACU_VAV_MID", "PACU_VAV_TOP"]
        acusandfloors = zip(ACUs, [1, 2, 3])

        # Create building object (and the associated zones)
        bldg = BuildingModel("18zones_ASHRAE901_OfficeMedium_STD2019_Denver", idf_filepath, epw_filepath,
                             output_folderpath, zonesandfloors, acusandfloors)

        # load the one-year simulation data
        bldg.load_simulation(training_data_filepath, "UTC")

        # set the nondispatchable load
        nd_load = bldg.simulation["Electricity:Building[Wh]"] / 1000
        nd_load.name = nd_load.name.replace("[Wh]", "[kWh]")
        bldg.set_nondispatchable_load(nd_load)

        # load the nn model
        nn_summary_filepath = os.path.abspath("../BuildingModel/Models/NN/TrainingSummary_NN.xlsx")
        NN_datetime = c.getbestmodel(nn_summary_filepath, nn_hyperparameters)
        nn_folderpath = os.path.join(os.path.dirname(nn_summary_filepath), f"NotSparse/NN_{nn_hyperparameters['nb_layers']}"
                                                                           f"layers_{nn_hyperparameters['nb_neurons']}neurons/"
                                                                           f"{NN_datetime}")
        # description of the neural network to be saved
        nn_description = f"{nn_hyperparameters['nb_layers']}layers_{nn_hyperparameters['nb_neurons']}neurons_each"
        bldg.load_nn(nn_folderpath, nn_description)

        # set the HVAC system and thermal requirements for each zone
        for z in bldg.zone_assets:
            z.set_HVAC(bldg, ems.T0)
        # compute the HVAC capacity of each PACU
        bldg.compute_PACU_capacity()

        ####################################
        ### Optimization problem
        ####################################

        # add the building to the EMS, /!\ for several buildings, create a list of buildings and add them with pd.concat
        ems.bldg_assets.loc[bldg.name] = bldg

        # Create the market conditions
        line_capacity = 1e3  # kW
        demand_charge = 0.4  # €/kW
        prices = np.transpose([[*[0.3] * 6, *[0.6] * 13, *[0.3] * 5], [0.2] * 24])[:ems.ts_ems[:-1].shape[0], :]  # €/kWh
        # prices go from midnight to 11pm
        elec_prices = pd.DataFrame(prices, index=ems.ts_ems[:-1], columns=["import_price", "export_price"])
        ems.set_market(demand_charge, elec_prices["import_price"], elec_prices["export_price"], line_capacity)

        # Tighten the bounds of the NN
        for b in ems.bldg_assets.values:
            is_parameter = [False] * len(bldg.nn_scaled_input_bounds)  # all the inputs are variable (not parameters)
            is_parameter[-1] = True  # outdoor temperature (or hour)
            is_parameter[-2] = True if len(bldg.nn_scaled_input_bounds) > bldg * 2 + 1  # outdoor temperature if the hour is there too
            b.adjust_bounds(ems.ts_ems, is_parameter)

        # Solve the optimization problem
        relu_formulations = {"nn_bigm": ReluBigMFormulation, "nn_ideal": ReluIdealFormulation,
                             "nn_extended1": ReluExtendedFormulation1, "nn_extended2": ReluExtendedFormulation2}
        relu_formulation = relu_formulations[
            thermodynamics_model] if thermodynamics_model in relu_formulations.keys() else None
        # ems.optimize(relu_formulation)
        ems.cvxpy_opti_formulation = ems.build_cvxpy(False, "milp")
        ems.set_parameters(ems.cvxpy_opti_formulation)
        ems.solve_cvxpy(ems.cvxpy_opti_formulation)
        ems.build_cvxpylayer("fixed_bin")

        ### compute the ex-post (simulation)
        ep = EnergyPlusSimulator()
        for b in ems.bldg_assets:
            b.idf_filepath_expost = modify_idf(b.idf_filepath, T0)
            simulationvariables = get_variable_list(b)
            # add the temperature setpoints to the list of variables
            for z in b.zone_assets[["Plenum" not in z.name for z in b.zone_assets]]:
                simulationvariables.append(f"Actuator,Zone Temperature Control,Cooling Setpoint,{z.name.upper()};")
                simulationvariables.append(f"Actuator,Zone Temperature Control,Heating Setpoint,{z.name.upper()};")
            # In this case, desired frequence is the desired frequency for the inputs of the simulation
            # it must be the one of the ems
            b.set_simulationvariables(simulationvariables, simulationfrequency=dt_sim, desiredfrequency=dt_ems)
            b.fullsimulation_expost, b.nb_warmupts_expost = ep.run_simulation(b, 3,
                                                                              ep.callback_temperature_control,
                                                                              b.idf_filepath_expost)
            b.simulation_expost = b.fullsimulation_expost.loc[~b.fullsimulation_expost["warmup"]]
            # set the dates as index
            b.simulation_expost.loc[:, "Timestep"] = b.simulation_expost["Timestep"].apply(
                c.create_datetime_from_Eplus_timestep)
            b.simulation_expost.set_index("Timestep", inplace=True)
            # Convert to Wh rather than J
            col = list(b.simulation_expost.columns)
            col_to_convert = [c for c in col if
                              (("energy" in c.lower() or "electricity" in c.lower() or "Heating:NaturalGas"
                                in c) and not ("rate" in c.lower()))]
            column_name_to_convert = c.flattenlist([c.getcolumnname(ctc, col) for ctc in col_to_convert])
            b.simulation_expost = c.convertJtoWh(b.simulation_expost, column_name_to_convert)
            # # Shift all the zone data which is not the temperature or the relative humidity
            # for col in b.simulation_expost.columns:
            #     if (sum([ping in col.lower() for ping in ["energy", "rate", "electricity", "naturalgas"]]) >= 1 and
            #             "Solar" not in col):
            #         b.simulation_expost[col] = b.simulation_expost[col].shift(-1)
            # compute the ex-post energy consumption
            b.simulation_expost = b.zonal_PACU_electricity_energy(b.simulation_expost)
            # set the expost Tin and P_hvac in each zone
            b.set_expost()

        # Train the cvxpylayer = tune the parameters of the neural network
        ems.train_cvxpylayer()


    ### Once the model has been trained
    for med in medoids_dates.index:
        # plot the results
        ncols = 5
        ems.plot_results(ncols, format="pdf", path=figure_folderpath, savefig=True)

        ### save the results
        save_results(results_summary_filepath, medoid_name, thermodynamics_model, ems)

