import operator
import warnings

import joblib
import numpy as np
import onnx
import os
import pandas as pd
import torch

from src.library.myomlt import OffsetScaling
from src.library.myomlt.io import load_onnx_neural_network
from src.library.myomlt.io.input_bounds import load_input_bounds


class BuildingModel:
    """
    BuildingModel is a class which contains the necessary information about an EnergyPlus existing model to simulate it.
    self.name: the name of the model
    self.idf_path: the path to the idf file
    self.weather_path: the path to the weather file
    self.output_path: the path to the folder where to save the output files of EnergyPlus
    self.zone_names: a list of the zone_names names
    self.ACUs: dictionary of the AirConditioningUnit names as values and the floor as keys
    self.zones_df: a dataframe with the name of the zones, the floor, the ACU and whether the zone is a plenum

    self.fullsimulation: the full simulation df with the warm start
    self.simulation: the simulation df without the warm start and the useless variables
    self.simulationvariables: a list of the variables to collect (the whole list)
    self.useless_variables: a list of the variables which are useless (i.e., always 0). This list is computed when
        the simulation is set (method: set_simulation).
    self.nb_batch255: the number of batch of 255 variables to collect (because of the limitation of ReadVarsESO to 256
        variables per batch. One is Date/Time which is always collected).
    self.simulationvariables_batch255: a list of lists of the variables to collect (the whole list) divided into
        batch of 255 variables.
    self.simulationfrequency: the hourly frequency of the simulation
    self.desiredfrequency: the desired frequency of the simulation (e.g., 1 for 1h, 4 for 15min, etc.)
    self.cop_polynom_degree: the degree of the polynom to approximate the COP relationship
    self.nb_warmupts: the number of time steps to use for sizing days simulation

    self.fullsimulation_expost: the full simulation df with the warm start for the ex-post simulation
    self.simulation_expost: the simulation df without the warm start and the useless variables for the ex-post simulation
    self.nb_warmupts_expost: the number of time steps to use for sizing days simulation for the ex-post simulation
    self.idf_filepath_expost: the path to the idf file for the ex-post simulation
    self.expost_cost: the cost of the ex-post simulation, cost for activating the HVAC following the temperature requirements

    self.nd_load: the non-dispatchable load of the building
    self.expected_results: a dataframe with the expected results of the optimization. I.e, the ems decisions.

    self.ml_model_doc: a string which contains the characteristics of the model. To be saved in the final excel
    """

    def __init__(self, name, idf_filepath, weather_path, output_folderpath, zonesandfloors: zip, ACUsandfloors: zip):
        """
        :param name: name of the simulation model
        :param idf_filepath: path to the idf file of the model
        :param weather_path: patht to the weather file
        :param output_folderpath: where to store the output files of the simulation
        :param zonesandfloors: a zip iterator object with the name of the zones and the floor
        :param ACUsandfloors: a zip iterator object with the name of the ACUs and the floor
        """
        zonesandfloors = list(zonesandfloors)
        ACUsandfloors = list(ACUsandfloors)
        self.name = name
        self.idf_filepath = idf_filepath
        self.weather_path = weather_path
        self.output_folderpath = output_folderpath
        self.zone_names = [z for z, _ in zonesandfloors]
        self.ACUs = pd.DataFrame(ACUsandfloors, columns=["ACU", "floor"])
        self.zones_df = pd.DataFrame(
            {"name": self.zone_names, "is_plenum": ["plenum" in z.lower() or "attic" in z.lower() for z in self.zone_names],
             "floor": [f for _, f in zonesandfloors],
             "ACU": [None] + self.ACUs["ACU"].to_list()})
        self.zones_df_no_plenum = self.zones_df.loc[~self.zones_df["is_plenum"], :]
        self.nbfloors = self.zones_df["floor"].max()
        self.zone_assets = pd.Series([Zone(z, f) for z, f in zonesandfloors], index=[z for z, _ in zonesandfloors])

        # Simulation variables
        self.fullsimulation = None
        self.simulation = None
        self.simulationvariables = None
        self.useless_variables = None
        self.nb_batch255 = None
        self.simulationvariables_batch255 = None
        self.simulationfrequency = None
        self.desiredfrequency = None
        self.cop_polynom_degree = None
        self.nb_warmupts = None

        # ex-post simulation
        self.fullsimulation_expost = None
        self.simulation_expost = None
        self.nb_warmupts_expost = None
        self.idf_filepath_expost = None
        self.hvac_expost_cost = None
        self.performance_loss = None

        # optimization variables
        self.nd_load = None
        self.expected_results = pd.DataFrame()
        self.expost_results = pd.DataFrame()

        # Machine Learning
        self.ml_model_doc = str()

        # neural network
        self.nn = None
        self.nn_onnx = None
        self.nn_scaler = None
        self.nn_offsetscaler = None
        self.nn_scaled_input_bounds = None
        self.nn_input_bounds = {}
        self.nn_flag = False

        # rcmodel
        self.rcmodel = None

    def set_simulationvariables(self, simulationvariables, simulationfrequency, desiredfrequency=1,
                                cop_polynom_degree=3):
        self.simulationvariables = simulationvariables
        self.nb_batch255 = len(simulationvariables) // 255
        self.simulationvariables_batch255 = (
                [simulationvariables[i * 255:(i + 1) * 255] for i in range(self.nb_batch255)]
                + [simulationvariables[self.nb_batch255 * 255:]])
        self.simulationfrequency = simulationfrequency
        self.desiredfrequency = desiredfrequency
        self.cop_polynom_degree = cop_polynom_degree
        self.nb_warmupts = None

    def set_nb_warmupts(self, n: int):
        self.nb_warmupts = n

    def set_simulations(self, fullsimulation: pd.DataFrame):
        """
        The input must be a full simulation with the warm start. The warm start is removed and the useless variables.
        :param fullsimulation:
        :return:
        """
        # Look for useless variables
        useless_variables = []
        for name, series in fullsimulation.items():
            if series.min() == series.max() and series.min() == 0:
                useless_variables.append(name)
        self.useless_variables = useless_variables
        simvar = fullsimulation.columns
        self.simulationvariables = [v for v in simvar if v not in useless_variables]
        self.fullsimulation = fullsimulation.loc[:, self.simulationvariables]
        # nb_warmupts must have been defined beforehand
        self.simulation = self.fullsimulation.loc[~self.fullsimulation["warmup"], :]

    def save_simulation(self, path, save_fullsimulation=False):
        """
        Save the full simulation in an excel and hdf file
        :param path: path is a path containing the name of the file without extension (e.g., "Model3/eplusout")
        :param save_fullsimulation: if True, save the full simulation (with the warm start) else, remov the warm_start
        :return:
        """
        df = self.fullsimulation if save_fullsimulation else self.simulation
        df.to_excel(path + ".xlsx", index=True, header=True, na_rep="NaN", freeze_panes=(1, 0))
        df.to_csv(path + ".csv", index=True, header=True, na_rep="NaN")
        df.to_hdf(path + ".h5", index=True, key="/df", mode="w", nan_rep="NaN")

    def load_simulation(self, filepath_hdf: str, tz: str = "UTC"):
        """
        load the simulation from the path
        :param path: path to the hdf file
        :return:
        """
        with pd.HDFStore(filepath_hdf, 'r') as store:
            self.simulation = pd.read_hdf(store, key="/df", mode="r")
        self.simulation.index = pd.DatetimeIndex(self.simulation.index, tz=tz)

    def set_nondispatchable_load(self, load: pd.Series):
        """
        Set the non-dispatchable load of the building in kW
        :param load: pd.DataFrame with the non-dispatchable
        :return:
        """
        self.nd_load = load


    def load_rcmodel(self, filepath):
        self.rcmodel = torch.load(filepath, map_location=torch.device('cpu'))

    def load_nn(self, nn_model_folder, ml_model_doc=None):
        """

        :param nn_model_folder: folder where to load the nn model
        :param ml_model_doc: a string which contains the characteristics of the model. To be saved in the final excel
        :return:
        """
        ### PATHS
        scaler_path = os.path.join(nn_model_folder, "scaler.pkl")  # path of the scaler file
        nn_onnx_path = os.path.join(nn_model_folder, "model.onnx")  # path of the nn file
        nn_bounds_path = os.path.join(nn_model_folder, "Bounds.json")  # path of the nn bounds file

        ### LOAD SCALERS & MODEL
        scalers_scikit = joblib.load(scaler_path)
        scalers = OffsetScaling(offset_inputs=scalers_scikit["input"].mean_,
                                factor_inputs=scalers_scikit["input"].scale_,
                                offset_outputs=scalers_scikit["output"].mean_,
                                factor_outputs=scalers_scikit["output"].scale_)
        nn_scaled_input_bounds = load_input_bounds(nn_bounds_path)
        nn_scaled_input_bounds_array = np.array([[e1, e2] for e1, e2 in nn_scaled_input_bounds.values()])

        nn_onnx = onnx.load(nn_onnx_path)  # load the model (NN, RF...)
        onnx.checker.check_model(nn_onnx)  # check that the model is well formed
        nn = load_onnx_neural_network(nn_onnx, scalers, nn_scaled_input_bounds)  # load the model (NN, RF...)
        self.nn = nn
        self.nn_onnx = nn_onnx
        self.nn_offsetscaler = scalers
        self.nn_scaler = scalers_scikit
        self.nn_scaled_input_bounds = nn_scaled_input_bounds
        self.nn_input_bounds = pd.DataFrame({"lb": scalers_scikit["input"].inverse_transform(
            nn_scaled_input_bounds_array[:, 0].reshape(1, -1)).flatten(),
                                             "ub": scalers_scikit["input"].inverse_transform(
                                                 nn_scaled_input_bounds_array[:, 1].reshape(1, -1)).flatten()},
                                            index=scalers_scikit["input"].feature_names_in_)
        self.nn_flag = True
        self.ml_model_doc = ml_model_doc

    def adjust_bounds(self, ts, is_parameter):
        """
        Methods to adjust the bounds of the neural network. The bounds are adjusted to be as tight as possible on the
        parameters.
        :param ts: the datetime index of the considered ts
        :param is_parameter: a boolean iterable which indicates if the feature is a parameter or not
        :return:
        """
        if self.nn_flag:
            ### Deal with the parameters then the variables
            for i, features_n in enumerate([is_parameter, list(map(operator.not_, is_parameter))]):
                features = self.nn_input_bounds.index[features_n]
                # we tighten the bounds to the specific period for the parameters
                # we make sure to get the extreme values over the database for the variables
                df = self.simulation.loc[ts, features] if i == 0 else self.simulation.loc[:, features]
                # If there is the electricity energy, we add a negative sign to cooling electricity
                if any(["Electricity Energy" in f for f in features]):
                    zonal_sensible_heating_rate = self.simulation.filter(axis=1,
                                                                        like="Zone Air System Sensible Heating Rate")
                    zonal_sensible_cooling_rate = self.simulation.filter(axis=1,
                                                                        like="Zone Air System Sensible Cooling Rate")
                    df.loc[:, ["Air System Electricity Energy" in c for c in df.columns]] = df.filter(
                        axis=1, like="Air System Electricity Energy").where(
                        zonal_sensible_heating_rate.values > zonal_sensible_cooling_rate.values,
                        -df.filter(axis=1, like="Air System Electricity Energy").values
                    ).div(1000)
                # to avoid an error because of a name change (can be suppressed later) and features_df be rename features
                # to be deleted later once new NN have been trained after April 1st 2024
                df.columns = features  # [f.split("[Wh]")[0] if "Zone PACU Electricity Energy," in f else f for f in df.columns]
                lb = df.min(axis=0)
                ub = df.max(axis=0)
                self.nn_input_bounds.loc[features, "lb"] = lb
                self.nn_input_bounds.loc[features, "ub"] = ub
            ### Deal with the variables
            tmp = pd.DataFrame(self.nn_scaler["input"].transform(self.nn_input_bounds.transpose()))
            self.nn_scaled_input_bounds = dict(zip(tmp.columns,
                                                   [tuple(tmp.values.transpose()[i]) for i in range(tmp.shape[1])]))
            self.nn = load_onnx_neural_network(self.nn_onnx, self.nn_offsetscaler, self.nn_scaled_input_bounds)
        else:
            warnings.warn("The neural network has not been loaded yet. The bounds cannot be adjusted.")

    def zonal_PACU_electricity_energy(self, df):
        """
        Compute the electrical consumption of the PACU associated to each zone.
        :param df: the dataframe containing the simulation results which needs to be enriched with the
            "Zone PACU Electricity Energy,{z['name']}" columns. E.g., df = bldg.simulation or bldg.simulation_expost
        :return: the enriched df
        """
        acu_group = self.zones_df_no_plenum.groupby("ACU")
        ttl_zone_vent = pd.DataFrame(columns=self.ACUs, index=df.index)
        for acu, g in acu_group:
            zone_mech_names = [f'Zone Mechanical Ventilation Mass Flow Rate,{g["name"]}' for _, g in g.iterrows()]
            ttl_zone_vent[acu] = df.loc[:, zone_mech_names].sum(axis=1)

        # To avoid fragmenting the dataframe, we store the new columns in a temporary dataframe then we concatenate it
        tmp_df = pd.DataFrame(index=df.index, columns=[f"Zone PACU Electricity Energy,{z['name']}[Wh]" for _, z in
                                                       self.zones_df_no_plenum.iterrows()])
        for i, z in self.zones_df_no_plenum.iterrows():
            tmp_df[f"Zone PACU Electricity Energy,{z['name']}[Wh]"] = (
                    (df[f"Zone Mechanical Ventilation Mass Flow Rate,{z['name']}"] / ttl_zone_vent[
                        z['ACU']]).fillna(0)
                    * (df[f"Air System DX Cooling Coil Electricity Energy,{z['ACU']}[Wh]"]
                       + 0.81 * df[f"Air System NaturalGas Energy,{z['ACU']}[Wh]"]
                       + df[f"Air System Fan Electricity Energy,{z['ACU']}[Wh]"])
                    + df[f"Heating Coil Electricity Energy,{z['name']} VAV BOX REHEAT COIL[Wh]"]
            )
        df = pd.concat([df, tmp_df], axis=1)

        return df

    def compute_PACU_capacity(self):
        """
        Compute the capacity for each ACU
        """
        acu_group = self.zones_df_no_plenum.groupby("ACU")
        self.ACU_capacity = pd.Series(index=self.ACUs["ACU"], dtype=float)
        for acu, df in acu_group:
            self.ACU_capacity[acu] = self.simulation.loc[:, f"Air System Electricity Energy,{acu}[Wh]"].max() / 1000

    def set_expost(self):
        """
        Set the ex-post simulation for each zone in the building
        """
        col_names = pd.MultiIndex.from_product([self.zones_df_no_plenum["name"], ['Tin', 'P_hvac']],
                                               names=['zone', 'variable'])
        for z in self.zone_assets:
            if z.controlled:
                cols = [f"Zone Mean Air Temperature,{z.name}", f"Air System Electricity Energy,{z.acu}[Wh]"]
                # rows of interest are after warmup and over the scheduling period
                mask = ~self.simulation_expost["warmup"] & \
                          (z.expected_results.index[0] <= self.simulation_expost.index) & \
                            (self.simulation_expost.index <= z.expected_results.index[-1])
                df = self.simulation_expost.where(mask).dropna()
                z.expost_results["Tin"] = df[cols[0]]
                z.expost_results["P_hvac"] = df[cols[1]]
                if self.expost_results.empty:
                    self.expost_results = pd.DataFrame(index=z.expost_results.index, columns=col_names)
                self.expost_results[z.name] = z.expost_results
                

    def compute_expost_cost(self, ts, market):
        """
        Compute the ex-post cost of the HVAC ONLY !!!

        :param ts: The DatetimeIndex of the ems decisions (must have a freq attribute)
        :param market: The prices of the electricity in €/kWh
        """
        # TODO: Change the way the cost is computed. Output the import and export cost from EnergyPlus
        hvac_powers = [f"Zone PACU Electricity Energy,{z.name}[Wh]" for z in self.zone_assets if z.controlled]
        ts_sim = (self.simulation_expost.index >= ts[0]) & (self.simulation_expost.index < ts[-1])
        agg_hvac_power = self.simulation_expost.loc[ts_sim, hvac_powers].sum(axis=1).resample(ts.freq).sum()
        self.hvac_expost_cost = 0
        for t in ts[:-1]:
            price = market.prices_import.loc[t] if agg_hvac_power.loc[t] >= 0 else market.prices_export.loc[t]
            self.hvac_expost_cost += agg_hvac_power[t] * price / 1000
        self.hvac_expost_cost += max(0, *agg_hvac_power) * market.demand_charge

    @property
    def temperature_mae(self):
        return sum([z.temperature_mae for z in self.zone_assets if z.controlled]) / len(self.zone_assets)

    @property
    def temperature_max_e(self):
        return max([z.temperature_max_e for z in self.zone_assets if z.controlled])

    @property
    def controlled_zone_names(self):
        return [z.name for z in self.zone_assets if z.controlled]


class Zone:
    """
    Zone is a class which contains the necessary information about a zone in a building.
    self.name: str
        The name of the zone
    self.floor: int
        The floor of the zone
    self.controlled: bool
        Whether the zone is controlled or not. Is there an imposed temeprature range?
    self.acu: str
        The name of the ACU which controls the zone
    self.Tin0: float
        The initial temperature of the zone
    self.Tmax: pd.Series
        The maximum temperature of the zone
    self.Tmin: pd.Series
        The minimum temperature of the zone
    self.Ttgt: pd.Series
        The target temperature of the zone for optimization
    self.hvac_capacity: pd.Series
        The electricity hvac capacity of the zone in kWe
    self.Tin_sim: pd.Series
        The simulated temperature of the zone
    self.simulation: pd.DataFrame
        The full simulation of the zone
    self.expected_results: pd.DataFrame
        The expected results of the optimization in a dataframe containing the ems decisions: the indoor temperature
        profile and the HVAC power profile
    self.expost_results: pd.DataFrame
        The ex-post results of the optimization in a dataframe containing the indoor temperature profile and the HVAC
        power profile as returned by energy plus
    self.occupancy_weights: pd.Series
        The occupancy weights of the building used to penalize the temperature deviation in the obj function
    """

    def __init__(self, name: str, floor: int):
        self.name = name
        self.floor = floor
        self.controlled = False
        self.acu = None
        self.Tin0 = None
        self.Tmax = None
        self.Tmin = None
        self.Ttgt = None
        self.hvac_capacity = None
        self.Tin_sim = None
        self.simulation = None
        self.expected_results = pd.DataFrame()
        self.expost_results = pd.DataFrame()
        self.occupancy_weights = None

    def set_HVAC(self, bldg: BuildingModel, initial_datetime: pd.Timestamp):
        """
        Set the HVAC parameters of the zone
        :param bldg:
        :param initial_datetime:
        :return:
        """
        if initial_datetime.tz != bldg.simulation.index.tz:
            raise ValueError("The time zone of the initial_datetime and the simulation index must be the same.")
        # extract the row as a series thanks to squeeze (and not as a dataframe)
        zone_row = bldg.zones_df.loc[bldg.zones_df["name"] == self.name].squeeze()
        self.controlled = ~zone_row["is_plenum"]
        self.acu = zone_row["ACU"]
        self.Tin0 = bldg.simulation[f"Zone Mean Air Temperature,{self.name}"].loc[initial_datetime]
        if self.controlled:
            self.Tmax = bldg.simulation[f"Zone Thermostat Cooling Setpoint Temperature,{self.name}"]
            self.Tmin = bldg.simulation[f"Zone Thermostat Heating Setpoint Temperature,{self.name}"]
            # self.Tmax = pd.Series(index=bldg.simulation.index, data=29)
            # self.Tmin = pd.Series(index=bldg.simulation.index, data=16)
            self.hvac_capacity = bldg.simulation[f"Air System Electricity Energy,{self.acu}[Wh]"].max() / 1000
            date = bldg.simulation[f"Air System Electricity Energy,{self.acu}[Wh]"].idxmax()
            # print(f"Max HVAC capacity of {self.name} occurs on {date}")

    def set_target_temperature(self, Ttgt: pd.Series):
        """
        Set the target temperature of the zone
        :param Ttgt: pd.Series with the target temperature
        :return:
        """
        self.Ttgt = Ttgt

    def set_simulation_results(self, simulation: pd.DataFrame):
        self.simulation = simulation
        self.Tin_sim = simulation[f"Zone Mean Air Temperature,{self.name}"]

    def set_occupancy_weights(self, weights: pd.Series):
        self.occupancy_weights = weights

    @property
    def temperature_mae(self):
        return (self.expected_results["Tin"] - self.expost_results["Tin"]).abs().mean()

    @property
    def temperature_max_e(self):
        return (self.expected_results["Tin"] - self.expost_results["Tin"]).abs().max()


class PhysicsBasedBuildingEquation:
    """
        class to contain the physics-based equation modelling the building temperature dynamics.

        Parameters
        ----------
        C : float
            Thermal capacitance of building (kWh/Degree C)
        R : float
            Thermal resistance of building to outside environment(Degree C/kW)
        COP_heating : float
            Coefficient of performance of the heat pump (N/A)
        COP_cooling : float
            Coefficient of performance of the chiller (N/A)
        deltat: float
            Time interval after which system is allowed to change decisions (h)
    """

    def __init__(self, C, R, COP_heating, COP_cooling, deltat):
        self.C = C
        self.R = R
        self.COP_heating = COP_heating
        self.COP_cooling = COP_cooling
        self.alpha = (1 - (deltat / (R * C)))
        self.beta = (deltat / C)
        self.gamma = deltat / (R * C)

    def predict(self, Tin_pts, Ta_pts, p_heating_pts, p_cooling_pts):
        """
        Function to predict the temperature inside the building at time t+1 given the temperature at time t,
        the ambient temperature and the power consumed by the HVAC system

        Parameters
        ----------
        Tin_pts : numpy.ndarray             pts = Previous Time Step
            Temperature inside the building at time t-1 (Degree C)
        Ta_pts : numpy.ndarray
            Ambient temperature at time t-1 (Degree C)
        p_heating_pts : numpy.ndarray
            Power consumed by the heating system at time t-1 (kW)
        p_cooling_pts : numpy.ndarray
            Power consumed by the cooling system at time t-1 (kW)
        Returns
        -------
        Tin : float
            Temperature inside the building at time t (Degree C)
        """
        Tin = (self.alpha * Tin_pts + self.beta * (self.COP_heating * p_heating_pts - self.COP_cooling * p_cooling_pts)
               + self.gamma * Ta_pts)
        return Tin
