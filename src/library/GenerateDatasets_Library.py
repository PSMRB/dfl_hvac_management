
from collections import OrderedDict
import joblib
import numpy as np
import os
import pandas as pd
import plotly
import plotly.graph_objects as go
from eppy.modeleditor import IDF
from sklearn import linear_model, preprocessing
# Homemade includes
import src.library.Common as c
import src.library.Plot as Plot
import src.library.EnergyPlusAPI as ep


def get_cop(df, save_results, model_path=None, identifier=None, polynom_degree: int = 1):
    """
    Function used in file GenerateDatasets.py to generate the linear regression based on the yearly simulation.
    The latest part compute the

    :param df: is the dataframe containing the data of the simulation (e.g. OutputVariables.xlsx)
    :param save_results: Bool to indicate if figures and regression parameters must be saved or not
    :param filepath: Filepath where to save the regression parameters (supposed to be with the OutputVariables.xlsx)
    :param identifier: Identifier of the COP (for the garage or for the living space)
    :return: a fig object from plotly and the reg object with scores
    """

    col = list(df.columns)
    heat_coil_elec_rate = np.array(df[c.getcolumnname("Heating Coil Electricity Rate [W](Hourly)", col)], ndmin=2).reshape(-1, 1)
    heat_coil_heat_rate = np.array(df[c.getcolumnname("Heating Coil Heating Rate [W](Hourly)", col)], ndmin=2).reshape(-1, 1)
    heating_on = heat_coil_elec_rate > 0
    heat_coil_cop = (heat_coil_heat_rate[heating_on] / heat_coil_elec_rate[heating_on]).reshape(-1, 1)
    cool_coil_elec_rate = np.array(df[c.getcolumnname("Cooling Coil Electricity Rate [W](Hourly)", col)], ndmin=2).reshape(-1, 1)
    cool_coil_heat_rate = np.array(df[c.getcolumnname("Cooling Coil Total Cooling Energy [Wh](Hourly)", col)], ndmin=2).reshape(-1, 1)
    cooling_on = cool_coil_elec_rate > 0
    cool_coil_cop = (cool_coil_heat_rate[cooling_on] / cool_coil_elec_rate[cooling_on]).reshape(-1, 1)
    outdoor_temp = np.array(df['Environment:Site Outdoor Air Drybulb Temperature [C](Hourly)'], ndmin=2).reshape(-1, 1)
    living_humidity = np.array(df[c.getcolumnname("Zone Air Relative Humidity [%](Hourly)", col)], ndmin=2).reshape(-1, 1)
    # indoor_temp = np.array(df['THERMAL ZONE LIVING SPACE:Zone Mean Air Temperature [C](Hourly)'], ndmin=2).reshape(-1, 1)

    # Container to be returned by the function
    Figs = []
    Regs = []
    Metrics = []
    P_hat = []

    for i, n in enumerate(["heating", "cooling"]):
        poly = preprocessing.PolynomialFeatures(degree=polynom_degree)
        reg = linear_model.LinearRegression()
        if n == "heating":
            # X = np.concatenate([outdoor_temp[heating_on].reshape(-1, 1), outdoor_wet_temp[heating_on].reshape(-1, 1), indoor_temp[heating_on].reshape(-1, 1),
            #                     heat_coil_elec_rate[heating_on].reshape(-1, 1)
            #                     ], axis=1)
            # After tests: only those two variables are relevant
            X = poly.fit_transform(np.concatenate([outdoor_temp[heating_on].reshape(-1, 1),
                                heat_coil_elec_rate[heating_on].reshape(-1, 1)
                                ], axis=1))
            y = heat_coil_cop

            reg.fit(X, y)
            y_hat = reg.predict(X)[:, 0]
            # get the vector for the full simulation (including sizing and cooling times where y_hat = 0)
            p_hat_year = np.zeros(df.shape[0])
            count = 0  # counter
            for h, b in enumerate(heating_on):
                if b:
                    p_hat_year[h] = heat_coil_heat_rate[h]/y_hat[count]
                    count += 1
            df_plot = pd.DataFrame({"x": X[:, 1], "y": X[:, 2], "z": y_hat, "z_tar": heat_coil_cop[:, 0]})
        else: # n == "cooling":
            # X = np.concatenate([outdoor_temp[cooling_on].reshape(-1, 1), outdoor_wet_temp[cooling_on].reshape(-1, 1), indoor_temp[cooling_on].reshape(-1, 1),
            #                     cool_coil_elec_rate[cooling_on].reshape(-1, 1)
            #                     ],
            #                    axis=1)
            # After tests: only those two variables are relevant
            X = poly.fit_transform(np.concatenate([outdoor_temp[cooling_on].reshape(-1, 1),
                                cool_coil_elec_rate[cooling_on].reshape(-1, 1),
                                living_humidity[cooling_on].reshape(-1, 1)
                                ],
                               axis=1))
            y = cool_coil_cop
            reg.fit(X, y)
            y_hat = reg.predict(X)[:, 0]
            # get the vector for the full simulation (including sizing and cooling times where y_hat = 0)
            p_hat_year = np.zeros(df.shape[0])
            count = 0
            for h, b in enumerate(cooling_on):
                if b:
                    p_hat_year[h] = cool_coil_heat_rate[h]/y_hat[count]
                    count += 1
            df_plot = pd.DataFrame({"x": X[:, 1], "y": X[:, 2], "z": y_hat, "z_tar": cool_coil_cop[:, 0]})

        r_squared = reg.score(X, y)
        mae = np.mean(np.abs(y-y_hat))
        mse = np.mean((y-y_hat)**2)
        metrics = {"R2": r_squared, "MAE": mae, "MSE": mse}

        print("Coefficients:\t", reg.coef_)
        print("Intercept:\t", reg.intercept_)
        print("R\N{SUPERSCRIPT TWO}:\t",  r_squared)
        print("MAE:\t", mae)
        print("MSE:\t", mse)

        # plot of the model versus target
        fig = go.Figure(data=[go.Scatter3d(x=df_plot["x"], y=df_plot["y"], z=df_plot["z"], mode='markers',
                                           marker=dict(size=3, opacity=0.9), name="Lin. Reg."),
                                go.Scatter3d(x=df_plot["x"], y=df_plot["y"], z=df_plot["z_tar"], mode='markers',
                                           marker=dict(size=3, opacity=0.9), name="Real COP")
                              ])
        fig.update_layout(title=f'COP fit', scene=dict(xaxis_title="Outdoot Temp. [°C]", yaxis_title="Power [W]",
                                                                   zaxis_title="COP"))
        if not save_results:  # without the if, the plot is displayed twice
            fig.show()

        if save_results:
            # save 3D figure of the linear regression
            fig_path = os.path.join(model_path, f"Figures/COP/cop_reg_{identifier}_{n}.html")
            lr_path = os.path.join(model_path, f"Models/COP/cop_reg_{identifier}_{n}.pkl")
            c.createdir(c.path_up(fig_path, 2))  # create the Figures folder if it doesn't exist
            c.createdir(c.path_up(fig_path, 1))  # create the COP folder if it doesn't exist
            plotly.offline.plot(fig, filename=fig_path)

            # Prepare the dataframes to be saved
            df_metrics = pd.DataFrame(metrics.values(), index=metrics.keys(), columns=["Metrics"])
            # dic_reg = {"outdoor_temp_coef": reg.coef_[0, 0], "elec_rate_coef": reg.coef_[0, 1], "intercept": reg.intercept_[0]}
            # df_reg = pd.DataFrame(np.array(list(dic_reg.values())).reshape(1, 3), columns=dic_reg.keys(), index=[0])

            # create a new sheet in OutputVariablesSimulation.xlsx to store the COP data
            filepath = os.path.join(model_path, f"Datasets/OutputVariablesSimulation.xlsx")
            with pd.ExcelWriter(filepath, mode='a', engine='openpyxl') as writer:
                df_metrics.to_excel(writer, sheet_name=f'cop_{identifier}_{n}', startrow=5)

            # save the regression model for the COP
            c.createdir(c.path_up(model_path, 2))  # create the Models folder if it doesn't exist
            c.createdir(c.path_up(model_path, 1))  # create the COP folder if it doesn't exist
            joblib.dump(reg, model_path)

        Figs.append(fig)
        Regs.append(reg)
        Metrics.append(metrics)
        P_hat.append(p_hat_year)

    return Figs, Regs, Metrics, P_hat


def renameeplusoutput(columns):
    """
    /!\ This function is only adapted for my model
    Function to rename the columns of the dataframe df to identify the thermal zone_names.
    Especially for the HVAC system.
    :param columns: a list of dataframe column names
    :return: the list of column names modified
    """
    columns = list(columns)
    new_col = []
    # rename the columns of the HVAC system
    for c in columns:
        # remove useless thermal zone information
        if "THERMAL ZONE" in c:
            new_col.append(c.replace("THERMAL ZONE", ""))
        elif "1 SPD DX HTG COIL" in c:
            substrings = c.split(":")
            if substrings[0][-1] == "4":
                p1 = "GARAGE"
            elif substrings[0][-1] == "3":
                p1 = "LIVING SPACE"
            new_col.append(p1 + ":" + substrings[1])
        elif "UNITARY" in c:
            substrings = c.split(":")
            if substrings[0][-1] == "1":
                p1 = "GARAGE"
            elif substrings[0][-1] == "T" or substrings[0][-1] == "L":
                p1 = "LIVING SPACE"
            else:
                raise ValueError("Unknown thermal zone HVAC")
            new_col.append(p1 + ":" + substrings[1])
        elif "FAN" in c:
            substrings = c.split(":")
            if substrings[0][-1] == "6":
                p1 = "GARAGE"
            elif substrings[0][-1] == "5":
                p1 = "LIVING SPACE"
            else:
                raise ValueError("Unknown thermal zone FAN")
            new_col.append(p1 + ":" + substrings[1])
        else:
            new_col.append(c)

    return new_col


def splitintozones(df, listofzones):
    """
    Function which splits the dataframe into N dataframes: one for N thermal zone_names
    :param df: the df which needs to be split
    :param listofzones: the list of thermal zone_names labels (e.g. ["LIVING", "GARAGE"])
    :return: a list of N dataframes
    """
    shared_columns = [c for c in df.columns if not any(z in c for z in listofzones)]
    shared_data = df.filter(items=shared_columns)
    df_l = []
    for z in listofzones:
        tmp_df = pd.concat([shared_data, df.filter(like=z)], axis=1)
        df_l.append(tmp_df)

    return df_l


def get_variable_list(bldg: c.BuildingModel):
    # the variables without the zone name
    variables = [
        "OutputVariable,Site Outdoor Air Drybulb Temperature,ENVIRONMENT",
        "OutputVariable,Site Outdoor Air Wetbulb Temperature,ENVIRONMENT",
        "OutputVariable,Site Outdoor Air Relative Humidity,ENVIRONMENT",
        "OutputVariable,Site Direct Solar Radiation Rate per Area,ENVIRONMENT",
        "OutputVariable,Site Diffuse Solar Radiation Rate per Area,ENVIRONMENT",
        "OutputVariable,Site Wind Speed,ENVIRONMENT",
        "OutputMeter,Electricity:Facility",
        "OutputMeter,Electricity:Building",
        "OutputMeter,Electricity:HVAC",
        "OutputMeter,Heating:Electricity",
        "OutputMeter,Heating:NaturalGas",
        "OutputMeter,Cooling:Electricity",
    ]

    # the variables with the zone name
    for _, z in bldg.zones_df.iterrows():
        n = z["name"]
        if True:  # "_bot" in n or "First" in n:
            variables.extend([
                f"OutputVariable,Zone Air Relative Humidity,{n}",
                f"OutputVariable,Zone Mean Air Temperature,{n}",
                f"OutputVariable,Zone Thermostat Cooling Setpoint Temperature,{n}",
                f"OutputVariable,Zone Thermostat Heating Setpoint Temperature,{n}",
                # The heating and cooling rate which actually reach the zone
                f"OutputVariable,Zone Air System Sensible Heating Rate,{n}",
                f"OutputVariable,Zone Air System Sensible Cooling Rate,{n}",
                # # Cooling:EnergyTransfer is similar to Zone Air System Sensible Cooling Rate
                # f"OutputMeter,Cooling:EnergyTransfer:Zone:{n}",
                # # Heating:EnergyTransfer is similar to Zone Air System Sensible Heating Rate
                # f"OutputMeter,Heating:EnergyTransfer:Zone:{n}",
                f"InternalVariable,Zone Air Volume,{n}",
            ])
            if not z["is_plenum"]:
                variables.extend([
                    f"OutputMeter,Electricity:Zone:{n}",
                    # Heating Energy and Heating Rate of the heating coil are the same for 1h ts
                    # And Heating Energy and Electricity Energy are equal

                    # f"OutputVariable,Zone Mechanical Ventilation No Load Heat Removal Energy,{n}",
                    # f"OutputVariable,Zone Mechanical Ventilation Cooling Load Increase Energy,{n}",
                    # f"OutputVariable,Zone Mechanical Ventilation Cooling Load Increase Due to Overheating Energy,{n}",
                    # f"OutputVariable,Zone Mechanical Ventilation Cooling Load Decrease Energy,{n}",
                    # f"OutputVariable,Zone Mechanical Ventilation No Load Heat Addition Energy,{n}",
                    # f"OutputVariable,Zone Mechanical Ventilation Heating Load Increase Energy,{n}",
                    # f"OutputVariable,Zone Mechanical Ventilation Heating Load Increase Due to Overcooling Energy,{n}",
                    # f"OutputVariable,Zone Mechanical Ventilation Heating Load Decrease Energy,{n}",
                    # f"OutputVariable,Zone Mechanical Ventilation Air Changes per Hour,{n}",
                    f"OutputVariable,Zone Mechanical Ventilation Mass Flow Rate,{n}"
                ])

    for a in bldg.ACUs["ACU"]:
        variables.extend([
            f"OutputVariable,Heating Coil Heating Energy,{a} HEATING COIL",
            f"OutputVariable,Heating Coil Electricity Energy,{a} HEATING COIL",
            # f"OutputVariable,Heating Coil Heating Rate,{z} VAV BOX REHEAT COIL",
            f"OutputVariable,Cooling Coil Total Cooling Energy,{a} HP COOLING COIL",
            f"OutputVariable,Cooling Coil Sensible Cooling Energy,{a} HP COOLING COIL",
            f"OutputVariable,Cooling Coil Electricity Energy,{a} HP COOLING COIL",
            f"OutputVariable,Air System Electricity Energy,{a}",

            f"OutputVariable,Air System Hot Water Energy,{a}",
                          f"OutputVariable,Air System Steam Energy,{a}",
                          f"OutputVariable,Air System Chilled Water Energy,{a}",
                          f"OutputVariable,Air System Electricity Energy,{a}",
                          # Air System NaturalGas Energy is similar to Air System Heating Coil NaturalGas Energy
                          f"OutputVariable,Air System NaturalGas Energy,{a}",
                          f"OutputVariable,Air System Water Volume,{a}",
                          f"OutputVariable,Air System Cooling Coil Total Cooling Energy,{a}",
                          # Air System Heating Coil Total Heating Energy is similar to Air System Heating Coil Electric
                          # + 0.81 * Air System Heating Coil NaturalGas Energy
                          f"OutputVariable,Air System Heating Coil Total Heating Energy,{a}",
                          f"OutputVariable,Air System Heating Coil Electricity Energy,{a}",
                          f"OutputVariable,Air System Heat Exchanger Total Heating Energy,{a}",
                          f"OutputVariable,Air System Heat Exchanger Total Cooling Energy,{a}",
                          f"OutputVariable,Air System Humidifier Total Heating Energy,{a}",
                          f"OutputVariable,Air System Evaporative Cooler Total Cooling Energy,{a}",
                          f"OutputVariable,Air System Desiccant Dehumidifier Total Cooling Energy,{a}",
                          # Fan Electricity Energy is similar to Air System Fan Electricity Energy
                          f"OutputVariable,Air System Fan Electricity Energy,{a}",
                          # f"OutputVariable,Air System Fan Air Heating Energy,{a}",
                          f"OutputVariable,Air System Heating Coil Hot Water Energy,{a}",
                          f"OutputVariable,Air System Cooling Coil Chilled Water Energy,{a}",
                          f"OutputVariable,Air System DX Heating Coil Electricity Energy,{a}",
                          f"OutputVariable,Air System DX Cooling Coil Electricity Energy,{a}",
                          f"OutputVariable,Air System Heating Coil NaturalGas Energy,{a}",
                          f"OutputVariable,Air System Heating Coil Steam Energy,{a}",
                          f"OutputVariable,Air System Humidifier Electricity Energy,{a}",
                          f"OutputVariable,Air System Evaporative Cooler Electricity Energy,{a}",
                          f"OutputVariable,Air System Desiccant Dehumidifier Electricity Energy,{a}",
                          f"OutputVariable,Air System Outdoor Air Mass Flow Rate,{a}",
                          f"InternalVariable,Intermediate Air System Main Supply Volume Flow Rate,{a}"])

    return variables

def set_time_step(hourly_ts_frequency, minimum_hourly_ts_frequency,
                  idf_filepath, idd_filepath="/Applications/EnergyPlus-22-2-0_arm64/Energy+.idd"):
    #                                                       /EnergyPlus-22-1-0/Energy+.idd
    """
    Set the number of time steps per hour for the simulation by modifying the idf file.
    Set the minimum system time step in minutes for the simulation by modifying the idf file.
    :param hourly_ts_frequency: the number of time steps per hour
    :param minimum_hourly_ts_frequency: the minimum system time step in minutes
    :param idf_filepath: the path to the idf file
    :param idd_filepath: the path to the idd file from E+. Must be adapted to the version of E+ used to run the idf file
    """
    IDF.setiddname(idd_filepath)
    idf = IDF(idf_filepath)
    convergence = idf.idfobjects["ConvergenceLimits"][0]
    convergence.Minimum_System_Timestep = 60 / minimum_hourly_ts_frequency  # must be in minutes
    ts = idf.idfobjects["Timestep"][0]
    ts.Number_of_Timesteps_per_Hour = hourly_ts_frequency
    idf.saveas(idf_filepath)


### principal function
def generate_datasets(buildingmodel: c.BuildingModel, save_fig: bool = False):
    ########## 0. Set the minimum time step ##########
    set_time_step(buildingmodel.simulationfrequency, buildingmodel.simulationfrequency,
                          buildingmodel.idf_filepath)
    ########## 1. Generate eplusout.csv ##########
    ep_sim = ep.EnergyPlusSimulator()
    container_df, nb_warmup_ts = ep_sim.run_simulation(buildingmodel, 3,
                                                       ep_sim.callback_no_actuation, buildingmodel.idf_filepath)
    buildingmodel.set_nb_warmupts(nb_warmup_ts)
    buildingmodel.set_simulations(container_df)
    buildingmodel.save_simulation(os.path.join(buildingmodel.output_folderpath, "eplusout"),
                                  save_fullsimulation=True)


    eplusout = c.create_clean_eplusoutcsv(buildingmodel, skipcsv=True)

    ########## 2. Save the data in excel, csv and hdf ##########
    buildingmodel.set_simulations(eplusout)
    folder_path = os.path.join(buildingmodel.output_folderpath, "Datasets")
    c.createdir(folder_path)
    buildingmodel.save_simulation(os.path.join(folder_path, "OutputVariablesEnergyPlus"), save_fullsimulation=False)

    ########## 3. Plot of the dataset ##########
    ### Resample the dataset to get hourly values
    # change the Date/Time column to consider the winter/summer time and make is DatetimeIndex object
    # right bound included. If ERROR: it is probably because the number of warm start time step in set wrong in
    # run_simulation
    buildingmodel.simulation.loc[:, "Timestep"] = pd.date_range("2017-01-01 00:00:00", "2018-01-01 00:00:00",
                                    freq=f"{60 / buildingmodel.simulationfrequency}min", tz=None)[1:-1]
    # Create a new column with the hour
    buildingmodel.simulation.insert(2, "Hour", buildingmodel.simulation["Timestep"].dt.hour)
    columns = list(buildingmodel.simulation.columns)
    vartosum = [col for col in columns if "[Wh]" in col]
    vartomean = [col for col in columns if (("Rate" in col) and not ("Radiation" in col))]
    vartofirst = [col for col in columns if not(col in vartosum + vartomean)]
    fct = []
    for col in columns:
        if col in vartosum:
            fct.append("sum")
        elif col in vartomean:
            fct.append("mean")
        elif col in vartofirst:
            fct.append("first")
        else:
            raise ValueError("Column not in any list")
    # resample the data to get the desired frequency (the Timestep becomes implicitly the index)
    buildingmodel.simulation = buildingmodel.simulation.resample(f"{60 / buildingmodel.desiredfrequency}min",
                                                                 on="Timestep").agg(OrderedDict(zip(columns, fct)))
    buildingmodel.simulation.drop(columns=["Timestep"], inplace=True)
    Plot.yearlyanalysis(buildingmodel, ["Cooling:Electricity[Wh]", "Heating:Electricity[Wh]"],
                        os.path.dirname(folder_path), save_fig)

    ########## 4. Compute the HVAC consumption of each zone ##########
    # Not needed for the small office building because each zone is equipped with its own HVAC system
    # buildingmodel.simulation = buildingmodel.zonal_PACU_electricity_energy(buildingmodel.simulation)

    ### Save OutputVariablesTraining
    final_filepath = os.path.join(folder_path, "OutputVariablesTraining")
    buildingmodel.save_simulation(final_filepath, save_fullsimulation=False)

    # Uncomment to open the excel and visualize results
    # os.system(f"open -a 'Microsoft Excel' {final_filepath}.xlsx")
    return buildingmodel.simulation

