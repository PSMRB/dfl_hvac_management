import cvxpy as cp
from datetime import datetime
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
from src.library.Building import BuildingModel
from src.library.Classes import Stopwatch
import src.library.Common as c
from src.library.EnergyPlusAPI import EnergyPlusSimulator
from src.library.GenerateDatasets_Library import get_variable_list
from src.library.Market import Market
from src.library.myomlt.block import OmltBlock
from src.library.myomlt.neuralnet.nn_formulation import ReluBigMFormulation, LeakyReluBigMFormulation
from src.library.myomlt.neuralnet.layer import InputLayer
import numpy as np
import os
import pandas as pd
import pickle
import pyomo.environ as pe
from pyomo.environ import Reals, NonNegativeReals, Binary
import re
import scienceplots
import torch
import warnings

# Set the default plt.show to 100 dpi but savefig to 600 to fit IEEE standards
plt.style.use(['science', 'ieee'])
plt.rcParams.update({'figure.dpi': '100', 'savefig.dpi': '600'})

# tbb.global_control(tbb.global_control.parameter.max_allowed_parallelism, 32)


class EMS:
    """
    EMS is a class which contains the necessary information about the Energy Management System of a building.
    self.T0: pd.Timestamp
        The initial datetime of the simulation
    self.T_ems: int
        The number of hours ahead in the EMS
    self.dt_ems: float
        The time step period of the EMS in hour (e.g. 0.25 for 15min)
    self.ts_ems: pd.DatetimeIndex (with the last time step being midnight next day)
        The datetime index of the EMS
    self.nb_ts_ems: int
        The number of time steps in the EMS including the midnight next day
    self.T_sim: int
        The number of hours in the simulation
    self.dt_sim: float
        The time step period of the simulation in hour (e.g. 0.25 for 15min)
    self.ts_sim: pd.DatetimeIndex (with the last time step being midnight next day)
        The datetime index of the simulation
    self.nb_ts_sim: int
        The number of time steps in the simulation including the midnight next day
    self.bldg_assets: pd.Series
        The pd.Series of the building assets (e.g., BuildingModel)
    self.nd_load_flag: bool
        The flag to indicate if the non-dispatchable load of the building is considered or not
    self.feasible: bool
        The flag to indicate if the optimization problem is feasible or not
    self.expected_results: pd.DataFrame
        The expected results of the optimization
    self.expected_obj_val: float
        The ex-post value of the problem (which might be different from the obj function of the optimization
        because it must be quadratic for CVXPYLayer but here, nothing forces that)
    self.expost_results: pd.DataFrame
        The ex-post results of the optimization
    self.expost_obj_val: float
        The ex-post value of the problem (which might be different from the obj function of the optimization
        because it must be quadratic for CVXPYLayer but here, nothing forces that)
    self.solver_status: dict
        The status of the solver. A dictionary which contains the following keys:
            - obj_val: float
                The value of the objective function
            - solver_engine: str
                The name of the solver engine
            - termination_condition: pyomo.opt.TerminationCondition
                The termination condition of the solver
            - options: dict
                The options of the solver
            - mip_gap: float
                The MIP gap
            - solving_time: float
                The solving time in seconds
            - status: pyomo.opt.SolverStatus
                The status of the solver
    self.market: Market
        The market object which contains the time-of-use prices and the feed-in prices
    self.cvxpy_opti_formulation: cp.Problem
        The formulation of the cvxpy optimization problem.
    self.cvxpylayer_opti_formulation: cp.Problem
        The formulation of the cvxpylayer optimization problem. Usually, it is the "fixed_bin" or the qp version.
    self.cvxpylayer: CvxpyLayer
        The cvxpylayer object which contains the cvxpy optimization problem
    self.cvxpylayer_init_param: list of torch.Tensor
        The initial parameters of the cvxpylayer object
    self.cvxpylayer_param_to_optimize: list of cvxpy.Parameter
        The parameters to optimize of the cvxpylayer object
    self.cvxpy_solution: dict
        The solution of the cvxpy optimization problem
    self.cvxpy_solution_names: list of strings
        The names of the cvxpy solution variables
    """

    def __init__(self):
        self.T0 = None

        self.T_ems = None
        self.dt_ems = None
        self.ts_ems = None
        self.nb_ts_ems = None

        self.T_sim = None
        self.dt_sim = None
        self.ts_sim = None
        self.nb_ts_sim = None

        self.bldg_assets = pd.Series(dtype=object)
        self.nd_load_flag = None
        self.feasible = None
        self.expected_results = pd.DataFrame()
        self.expected_obj_val = None
        self.expected_power_cost = None
        self.expected_temperature_penalty = None
        self.expost_results = pd.DataFrame()
        self.expost_power_cost = None
        self.expost_temperature_penalty = None
        self.solver_status = dict()

        self.cvxpy_opti_formulation = None
        self.thermal_model = None  # "rc" or "nn"
        self.relu_bin_formulation = None  # "miqp", "fixed_bin", "qp"
        self.thermal_model_target_name = None  # "Zone Mean Air Temperature(t+1)"
        # self.cvxpylayer_opti_formulation = None
        # self.cvxpylayer = None
        # self.cvxpylayer_init_param = None
        # self.cvxpylayer_param_to_optimize = None
        self.cvxpy_solution = None

        # market variables
        self.market = None

    # Function used by pickle to save the object
    def __getstate__(self):
        """
        Not all class attributes are saved in the pickle file. The cvxpy_ attributes are not saved because they are not
        serializable.
        Returns:

        """
        state = self.__dict__.copy()
        # remove or handle non-serializable attributes
        if isinstance(state["cvxpy_opti_formulation"], cp.Problem):
            param_dic = state["cvxpy_opti_formulation"].param_dict
            var_dic = state["cvxpy_opti_formulation"].var_dict
            state.pop("cvxpy_opti_formulation", None)
            state["cvxpy_opti_formulation"] = {"param_dic": param_dic, "variable_dic": var_dic}
        state.pop("bldg_assets", None)
        state.pop("market", None)
        state["solver_status"].pop("options", None)
        state["solver_status"].pop("status", None)
        return state

    @property
    def expost_obj_val(self):
        return self.expost_power_cost + self.expost_temperature_penalty


    @classmethod
    def build_ems(cls, day_date, thermal_model, paths, now,
                  cvxpy_opti_formulation, relu_relaxation_for_convex_opti,
                  warm_start, hyperparameters, snr):
        """
        Factory method to create the EMS object for the case study.

        :param day_date: the date of the day to simulate (string "dd/mm/yyyy")
        :param thermal_model: either "nn" or "rcmodel" or "spatialrcmodel"
        :param paths: a dictionary with the paths to the data
        :param now: the current date and time (string "yyyy-mm-dd_HH-MM-SS")
        :param cvxpy_opti_formulation: either None or a cvxpy formulation
        :param relu_relaxation_for_convex_opti: one of "miqp", "qp", "fixed_bin"
        :param warm_start: either "True", "False", "Noise", or "dfl"
        :param hyperparameters: a dictionary with the hyperparameters of the neural network or rc model
        :param snr: the signal to noise ratio (only if warm_start == "Noise")
        :param best_epoch: if not None, the epoch of the model to be used (int)

        N.B.: the cvxpy problem is formulated but the parameter values are not assigned.
        """
        ems = cls()

        # set the EMS times
        T0_date = day_date
        T0 = pd.Timestamp(T0_date, tz="UTC")  # starting datetime of the simulation
        # T0 = pd.Timestamp("01/01/2017", tz="UTC")
        T_ems = 24  # number of hours in the EMS
        dt_ems = 1  # time step period of the EMS in hour (e.g. 0.25 for 15min)
        T_sim = 24  # number of hours in the simulation
        dt_sim = 0.25  # time step period of the simulation in hour (e.g. 0.25 for 15min)
        ems.set_times(T0, T_ems, dt_ems, T_sim, dt_sim)

        ####################################
        ### Create the BuildingModel object
        ####################################

        # parameters
        zone_names = ['Attic', 'Core_ZN', 'Perimeter_ZN_1', 'Perimeter_ZN_2', 'Perimeter_ZN_3', 'Perimeter_ZN_4']
        floors = [1] * 6
        zonesandfloors = zip(zone_names, floors)
        # HVAC system names (in the same order as the zone_names)
        ACUs = [f"PSZ-AC:{i}" for i in range(1, 6)]
        acusandfloors = zip(ACUs, [1] * 5)

        # Create building object (and the associated zones)
        bldg = BuildingModel("6zones_ASHRAE901_OfficeSmall_STD2020_Denver_2006-2020", paths["idf_filepath"],
                             paths["epw_filepath"], paths["output_folderpath"], zonesandfloors, acusandfloors)

        # load the one-year simulation data
        bldg.load_simulation(paths["training_data_filepath"], "UTC")

        # set the nondispatchable load
        nd_load = bldg.simulation["Electricity:Building[Wh]"] / 1000
        nd_load.name = nd_load.name.replace("[Wh]", "[kWh]")
        bldg.set_nondispatchable_load(nd_load)

        # set the HVAC system and thermal requirements for each zone
        for z in bldg.zone_assets:
            z.set_HVAC(bldg, ems.T0)
            yearly_occ = c.tile_daily_pattern(bldg.simulation.index, pd.Series([0.1] * 8 + [1] * 11 + [0.3] * 5))
            z.set_occupancy_weights(yearly_occ)
            z.set_target_temperature(pd.Series(21, index=bldg.simulation.index))
        # compute the HVAC capacity of each PACU
        bldg.compute_PACU_capacity()

        ####################################
        ### Optimization problem
        ####################################

        # add the building to the EMS, /!\ for several buildings, create a list of buildings and add them with pd.concat
        ems.bldg_assets.loc[bldg.name] = bldg

        # Create the market conditions
        line_capacity = 1e4  # kW
        demand_charge = 0.4  # €/kW
        daily_prices = np.transpose([[*[0.3] * 6, *[0.6] * 13, *[0.3] * 5], [0.2] * 24])[:ems.ts_ems[:-1].shape[0],:]  # €/kWh
        yearly_import_prices = c.tile_daily_pattern(bldg.simulation.index, pd.Series(daily_prices[:, 0],
                                                                                     index=ems.ts_ems[:-1]))
        yearly_export_prices = c.tile_daily_pattern(bldg.simulation.index, pd.Series(daily_prices[:, 1],
                                                                                     index=ems.ts_ems[:-1]))
        # prices go from midnight to 11pm
        elec_prices = pd.concat({"import_price": yearly_import_prices, "export_price": yearly_export_prices}, axis=1)
        ems.set_market(demand_charge, elec_prices["import_price"], elec_prices["export_price"], line_capacity)

        ### Load thermal model (and build save_folderpath)
        # if we do not load from a previous training (best_epoch is None), load the best model from the historical warm-start
        if thermal_model == "nn":
            NN_tail_folderpath = os.path.dirname(re.split("/NN/|/cvxpylayer/", paths["initial_model_folderpath"])[1])
            cvxpylayer_model_save_folderpath = os.path.join(paths["cvxpylayer_models_folderpath"], NN_tail_folderpath,
                                                            now)
            if "dfl" in warm_start:
                with open(os.path.join(paths["initial_model_folderpath"], "summary.pkl"), 'rb') as file:
                    summary = pickle.load(file)
                with open(os.path.join(paths["initial_model_folderpath"], "weights.pkl"), 'rb') as file:
                    weights_df = pickle.load(file)
                    weights = weights_df.iat[summary['Best Epoch'] + 1, 0]
                with open(os.path.join(paths["initial_model_folderpath"], "biases.pkl"), 'rb') as file:
                    biases_df = pickle.load(file)
                    biases = biases_df.iat[summary['Best Epoch'] + 1, 0]
                # load the initial nn model that was trained on historical data and warm-started the dfl training
                nn_summary_filepath = os.path.abspath("data/SmallOffice/Models/NN/TrainingSummary_NN2.xlsx")
                NN_datetime = c.getbestmodel(nn_summary_filepath, hyperparameters)
                NN_tail_filepath = (f"NotSparse/NN_{hyperparameters['nb_layers']}layers_"
                                    f"{hyperparameters['nb_neurons']}neurons/{NN_datetime}")
                ito_model_folderpath = os.path.join(os.path.dirname(nn_summary_filepath), NN_tail_filepath)
                # description of the neural network to be saved
                nn_description = f"{hyperparameters['nb_layers']}layers_{hyperparameters['nb_neurons']}neurons_each"
                # load nn model
                bldg.load_nn(ito_model_folderpath, nn_description)
            else:
                # description of the neural network to be saved
                nn_description = f"{hyperparameters['nb_layers']}layers_{hyperparameters['nb_neurons']}neurons_each"
                # load nn model
                bldg.load_nn(paths["initial_model_folderpath"], nn_description)
                # get the weights and biases from the onnx model for the first iteration
                weights, biases = EMS.get_w_and_b_from_onnx(ems.bldg_assets, warm_start, snr)
            # Tighten the bounds of the NN
            for b in ems.bldg_assets.values:
                # all the inputs are variable (not parameters)
                is_parameter = [False] * len(bldg.nn_scaled_input_bounds)
                if "not_tight" not in warm_start:
                    # all the inputs after 2 * the number of controlled zones are parameters
                    nb_variables = 2 * len(b.controlled_zone_names)
                    is_parameter[nb_variables:] = [True] * len(is_parameter[nb_variables:])
                b.adjust_bounds(ems.ts_ems, is_parameter)
        elif thermal_model == "rcmodel":
            # folderpath to save the output
            cvxpylayer_model_save_folderpath = os.path.join(paths["cvxpylayer_models_folderpath"], "RC", now)
            # get the weights and biases from the pickle model for the first iteration
            if "dfl" in warm_start:  # load a model trained with dfl
                with open(os.path.join(paths["initial_model_folderpath"], "summary.pkl"), 'rb') as file:
                    summary = pickle.load(file)
                with open(os.path.join(paths["initial_model_folderpath"], "weights.pkl"), 'rb') as file:
                    weights_df = pickle.load(file)
                    weights = weights_df.iat[summary['Best Epoch'] + 1, 0]
                with open(os.path.join(paths["initial_model_folderpath"], "biases.pkl"), 'rb') as file:
                    biases_df = pickle.load(file)
                    biases = biases_df.iat[summary['Best Epoch'] + 1, 0]
            else:  # load the model pre-trained on historical data
                # load pytorch rcmodel
                bldg.load_rcmodel(os.path.join(paths["initial_model_folderpath"], "RCmodel.pth"))
                # get the weights and biases from the torch model for the first iteration
                weights, biases = EMS.get_w_and_b_from_rc_pkl(ems.bldg_assets, warm_start, hyperparameters["target"],
                                                              snr)
        elif thermal_model == "spatialrcmodel":
            # folderpath to save the output
            cvxpylayer_model_save_folderpath = os.path.join(paths["cvxpylayer_models_folderpath"], "spatialRC", now)
            # get the weights and biases from the pickle model for the first iteration
            if "dfl" in warm_start:  # load a model trained with dfl
                with open(os.path.join(paths["initial_model_folderpath"], "summary.pkl"), 'rb') as file:
                    summary = pickle.load(file)
                with open(os.path.join(paths["initial_model_folderpath"], "weights.pkl"), 'rb') as file:
                    weights_df = pickle.load(file)
                    weights = weights_df.iat[summary['Best Epoch'] + 1, 0]
                with open(os.path.join(paths["initial_model_folderpath"], "biases.pkl"), 'rb') as file:
                    biases_df = pickle.load(file)
                    biases = biases_df.iat[summary['Best Epoch'] + 1, 0]
            else:  # load the model pre-trained on historical data
                # load pytorch spatial rcmodel
                bldg.load_rcmodel(os.path.join(paths["initial_model_folderpath"], "RCmodel.pth"))
                # get the weights and biases from the torch model for the first iteration
                weights, biases = EMS.get_w_and_b_from_rc_pkl(ems.bldg_assets, warm_start, hyperparameters["target"],
                                                              snr)
        else:
            raise ValueError("The thermal model must be either 'nn' or '(spatial)rcmodel'.")
        c.createdir(cvxpylayer_model_save_folderpath, up=2)

        # Build the cvxpy formulation in the ems object
        if cvxpy_opti_formulation is None:
            if relu_relaxation_for_convex_opti is None:
                raise ValueError(
                    "Either the relu relaxation ('miqp', 'qp', 'fixed_bin') OR the thermal model ('rcmodel', 'nn')"
                    "must be provided to build the cvxpy formulation.")
            ems.cvxpy_opti_formulation = ems.build_cvxpy(relu_relaxation_for_convex_opti, thermal_model)
        else:
            ems.thermal_model = thermal_model
            ems.cvxpy_opti_formulation = cvxpy_opti_formulation

        return ems, weights, biases, cvxpylayer_model_save_folderpath


    def update_ems(self, day_date):
        # set the EMS times
        T0_date = day_date
        T0 = pd.Timestamp(T0_date, tz="UTC")  # starting datetime of the simulation
        T_ems = 24  # number of hours in the EMS
        dt_ems = 1  # time step period of the EMS in hour (e.g. 0.25 for 15min)
        T_sim = 24  # number of hours in the simulation
        dt_sim = 0.25  # time step period of the simulation in hour (e.g. 0.25 for 15min)
        self.set_times(T0, T_ems, dt_ems, T_sim, dt_sim)

        # update the initial temperature of the zones
        for b in self.bldg_assets:
            for z in b.zone_assets:
                z.Tin0 = b.simulation[f"Zone Mean Air Temperature,{z.name}"].loc[T0]


    def set_times(self, T0: pd.Timestamp, T_ems: int = None, dt_ems: float = None, T_sim: int = None,
                  dt_sim: float = None):
        """
        Initialize (or reset) the times of the EMS
        :param T0:
        :param T_ems:
        :param dt_ems:
        :param T_sim:
        :param dt_sim:
        """
        self.T0 = T0

        self.T_ems = T_ems
        self.dt_ems = dt_ems
        self.ts_ems = pd.date_range(start=self.T0, periods=self.T_ems / self.dt_ems + 1, freq=f"{self.dt_ems}H",
                                    tz=self.T0.tzinfo)
        self.nb_ts_ems = len(self.ts_ems)

        self.T_sim = T_sim
        self.dt_sim = dt_sim
        self.ts_sim = pd.date_range(start=self.T0, periods=self.T_sim / self.dt_sim + 1, freq=f"{self.dt_sim}H",
                                    tz=self.T0.tzinfo)
        self.nb_ts_sim = len(self.ts_sim)

    def set_market(self, demand_charge, prices_import, prices_export, line_capacity: float = 1e6):
        # check the datetimeindex of the pricesmatch the ems
        if not all([all([ts in prices_import.index, ts in prices_export.index]) for ts in self.ts_ems[:-1]]):
            raise IndexError("The datetime index of the prices do not contain all the EMS datetime index.")
        self.market = Market(demand_charge, prices_import, prices_export, line_capacity)

    @staticmethod
    def get_w_and_b_from_onnx(bldg_assets, warm_start, SNR, rg = False):
        """
        Function to get the weights and biases from the onnx file
        :param bldg_assets: pd.Series of BuildingModel (the attribute contained in EMS object)
        :return: two dictionaries containing the weights and biases of the cvxpylayer network with keys =
                building name and values = dictionary with keys = layer number and values = torch.Tensor
        """

        weights, biases = {}, {}
        for b in bldg_assets:
            bn = b.name
            weights[bn], biases[bn] = b.get_w_and_b_from_onnx(warm_start, SNR, rg)

        return weights, biases


    @staticmethod
    def get_w_and_b_from_rc_pkl(bldg_assets, warm_start, target_name, SNR):
        """
        Function to get the weights and biases from the rc model
        :param bldg_assets: pd.Series of BuildingModel (the attribute contained in EMS object)
        :return:
        """
        weights, biases = {}, {}
        for b in bldg_assets:
            bn = b.name
            weights[bn], biases[bn] = b.get_w_and_b_from_rc_pickle(warm_start, target_name, SNR)

        return weights, biases


    def build_cvxpy(self, relu_relaxation, thermal_model: str = "nn",
                    target_name="Zone Mean Air Temperature(t+1)"):
        """
        This function only writes the EMS of the building. The optimization problem is written with cvxpy.
        None of the parameters is given a value. They are created as cvxpy parameters but not given a value.
        :param relu_bin_formulation: the formulation of the optimization problem
                    - "miqp": the relu binaries are variables
                    - "fixed_bin": the relu binaries are parameters
                    - "qp": the relu binaries are continuous variables between 0 and 1
        :param thermal_model: a str indicating which model is used for the thermodynamics of the building
                    - "rcmodel" for the rc model
                    - "nn" for the nn model
        :return:
        """

        self.relu_bin_formulation = relu_relaxation
        self.thermal_model = thermal_model
        self.thermal_model_target_name = target_name

        def build_nn_formulation(relu_relaxation, C_additional=[]):
            """
            Function which takes a nn loaded with the following command:
                    load_onnx_neural_network(nn_onnx, scalers, nn_scaled_input_bounds)
            and build the MILP formulation out of it. It is based on the OMLT package in pyomo but adapted to cvxpy
            where there is no block concept.
            :param weights: iterable of weights for each layer. If None, the weights are inferred from the onnx model.
                            To specify the weights manually, structure is [building][layer] where building
                            is the name of the building and layer is the index (integer) of the layer. The
                            weight matrix must be of shape (nb of neurons in the previous layer, nb of neurons in the current layer)
                            (Y = X * W + b)
            :param biases: iterable of weights for each layer. If None, the biases are inferred from the onnx model.
                            To specify the biases manually, with structure [building][layer]. Must be of single dimension.
            :param relu_binaries: iterable of binaries for each layer, for each neuron. If None, the binaries are
                                    inferred from the onnx model. Else, they take the expression in relu_binaries.
                                    It can be used to set the binaries to 0 or 1 (or any other constant). It can
                                    also be used to define them as continuous variables. Then,additionnal constraints
                                    may be added to ensure define the bounds of the variables.
                                    Structure is [building][layer][neuron]
            :param C_additional: list of additional constraints to add to the formulation.
            :return: the constraints, the sets and the parameters for the MILP formulation
            """
            m = pe.ConcreteModel()
            m.S_buildings = pe.Set(initialize=S_buildings)

            nn_onnx, nn_layers = {}, {}  # variables used to read the nn
            nn_activations, nn_is_input_layer, nn_input_layer_idx = {}, {}, {}  # info on the nn
            ## Sets
            S_nn_layers, S_nn_inputs, S_nn_zs = {}, {}, {}  # sets of the formulation
            S_nn_zhats = S_nn_zs  # both sets are similar
            ## Parameters
            nn_weights, nn_biases = {}, {}  # to be optimized by DFL
            zhat_bounds, z_bounds = {}, {}  # bounds for the z and zhat variables
            # Parameter to store the value of the nn binaries if we used the fixed_bin formulation
            nn_relu_bin_dpp = {}
            ## Variables
            # nn_relu_bin is the binary variable for the ReLU (can be a parameter depending on the relu_bin_formulation)
            nn_relu_bin = {}
            # nn_zhats_dpp is here just to keep the problem DPP = no product of
            # two terms containing parameters
            # https://www.cvxpy.org/tutorial/advanced/index.html#disciplined-parametrized-programming
            nn_zhats_dpp = {}
            # other variables
            nn_inputs, nn_zs, nn_zhats, nn_unscaled_outputs = {}, {}, {}, {}
            ## Constraints
            # Equality constraint: indoor temperature/HVAC power = the first nn inputS
            # 1D list of length nb_zones * nb of ts
            C_input_nn_tin, C_input_nn_phvac = [], []
            # Equality constraint: T_amb = former last nn input, size = nb of ts
            C_input_nn_tamb = []
            # equality constraint: time step t = last nn input, size = nb of ts
            C_input_nn_t = []
            # scaling input layer: y = (x - mu) / sigma, size = nb of inputs * nb of ts
            C_input_scaling = []
            C_preactivation, C_hidden_layer = [], []  # preactivation and hidden layer constraints (MILP formulation)
            # each of the constraints list below is a 1D list of length output_layer_size * nb of ts
            C_output_unscale, C_output_nn = [], []  # constraints
            # 4 constraints per neuron in the hidden layer (ub and lb for z and zhat) * nb of ts
            C_bounds_nn = []  # bounds constraints
            # bounds (0, 1) for the binary variables if we use the LP relaxation
            C_bounds_relu_bin = []  # bounds constraints for the binary variables

            # for each building
            for bn in S_buildings:
                # declare the variables
                S_nn_inputs[bn], S_nn_zs[bn] = {}, {}  # Sets
                nn_weights[bn], nn_biases[bn] = {}, {}  # Parameters to be optimized by DFL
                zhat_bounds[bn], z_bounds[bn] = {}, {}  # Parameters for the bounds
                nn_activations[bn], nn_is_input_layer[bn], nn_input_layer_idx[bn] = {}, {}, None  # Info on the nn
                nn_zs[bn], nn_zhats[bn], nn_zhats_dpp[bn], nn_unscaled_outputs[bn], = {}, {}, {}, {}  # Variables
                nn_relu_bin[bn], nn_relu_bin_dpp[bn] = {}, {}  # Variables (or Param) for the ReLU

                # read the nn from onnx
                nn_onnx[bn] = self.bldg_assets[bn].nn
                layers_onnx = list(nn_onnx[bn].layers)

                def build_formulation(nnblock, bn):
                    nnblock.nn_omlt_milp_formulation = OmltBlock()
                    # Assumption: The first layer after the input layer has the same activation function as the rest
                    # of the hidden layers
                    for layer_onnx in layers_onnx:
                        if not isinstance(layer_onnx, InputLayer):
                            if layer_onnx.activation == "relu":
                                nnblock.nn_omlt_milp_formulation.build_formulation(ReluBigMFormulation(nn_onnx[bn]))
                                break
                            elif layer_onnx.activation == "leakyrelu":
                                nnblock.nn_omlt_milp_formulation.build_formulation(LeakyReluBigMFormulation(nn_onnx[bn],
                                                                                                            alpha=0.01))
                                break

                m.B_nn_omlt_milp_formulation = pe.Block(m.S_buildings, rule=build_formulation)

                layers_omlt = [l for l in m.B_nn_omlt_milp_formulation[bn].nn_omlt_milp_formulation.layer.values()]
                for l, (layer_onnx, layer_omlt) in enumerate(zip(layers_onnx, layers_omlt)):
                    # declare the parameters
                    zhat_bounds[bn][l], z_bounds[bn][l] = {}, {}  # declare the bounds
                    # declare the variables
                    nn_inputs[bn], nn_zs[bn][l], nn_zhats[bn][l], nn_zhats_dpp[bn][l] = {}, {}, {}, {}
                    nn_relu_bin[bn][l], nn_relu_bin_dpp[bn][l] = {}, {}

                    ### Info on the neural network
                    # activation for each layer
                    nn_activations[bn][l] = layer_onnx.activation
                    # if the layer is the input layer
                    nn_is_input_layer[bn][l] = isinstance(layer_onnx, InputLayer)
                    # index of the input layer (independent from l (could be out of the l loop))
                    nn_input_layer_idx[bn] = l if nn_is_input_layer[bn][l] else nn_input_layer_idx[bn]

                    ### Sets
                    # index set of the layers [0, 1, 2, ..., L-1], independent from l (could be out of the l loop)
                    S_nn_layers[bn] = range(len(layers_onnx))
                    # index set of the inputs for each layer, independent from l (could be out of the l loop)
                    S_nn_inputs[bn] = [i[0] for i in layers_onnx[nn_input_layer_idx[bn]].input_indexes]
                    # index set of the outputs for each layer
                    S_nn_zs[bn][l] = [i[0] for i in layer_onnx.output_indexes]

                    ### Parameters
                    # declare the bounds
                    for o in S_nn_zs[bn][l]:
                        if not nn_is_input_layer[bn][l]:  # input layer has no zhat
                            zhat_bounds[bn][l][o] = cp.Parameter((D_S_ts, 2), name=f"nn_layer_{l}_zhat_bound_{o}")
                            # if not cvxpylayer_formulation:
                            #     zhat_bounds[bn][l][o].value = np.array([layer_omlt.zhat[o].bounds for t in S_ts])
                        z_bounds[bn][l][o] = cp.Parameter((D_S_ts, 2), name=f"nn_layer_{l}_z_bound_{o}")
                        # if not cvxpylayer_formulation:
                        #     z_bounds[bn][l][o].value = np.array([layer_omlt.z[o].bounds for t in S_ts])

                    # weights and biases
                    if not isinstance(layer_onnx, InputLayer):  # input layer has no weights and biases
                        nn_weights[bn][l] = cp.Parameter((len(S_nn_zs[bn][l - 1]), len(S_nn_zs[bn][l])),
                                                         name=f"nn_layer_{l}_weights")
                        nn_biases[bn][l] = cp.Parameter(len(S_nn_zs[bn][l]), name=f"nn_layer_{l}_biases")

                    ### Variables
                    # input variables
                    for i in S_nn_inputs[bn]:
                        nn_inputs[bn][i] = cp.Variable((D_S_ts,), name=f"nn_layer_{l}_input_{i}")
                    # output and binary variables
                    for o in S_nn_zs[bn][l]:
                        if not nn_is_input_layer[bn][l]:
                            nn_zhats[bn][l][o] = cp.Variable((D_S_ts,), name=f"nn_layer_{l}_zhat_{o}")
                            nn_zhats_dpp[bn][l][o] = cp.Variable((D_S_ts,), name=f"nn_layer_{l}_zhat_long_{o}")
                        nn_zs[bn][l][o] = cp.Variable((D_S_ts,), name=f"nn_layer_{l}_z_{o}")
                        # descaled output variables
                        if l == S_nn_layers[bn][-1]:
                            nn_unscaled_outputs[bn][o] = cp.Variable((D_S_ts,), name=f"nn_output_descaled_{o}")
                        # binary variables for the relu
                        if nn_activations[bn][l] == "relu" or nn_activations[bn][l] == "leakyrelu":
                            match relu_relaxation:
                                case "miqp":
                                    # Binaries are variables
                                    nn_relu_bin[bn][l][o] = cp.Variable((D_S_ts,), boolean=True,
                                                                        name=f"nn_layer_{l}_binary_{o}")
                                case "fixed_bin":
                                    # binaries are parameters (fixed)
                                    # N.B.: there is a need of a variable to keep the formulation dpp
                                    nn_relu_bin_dpp[bn][l][o] = cp.Parameter((D_S_ts,),
                                                                             name=f"nn_layer_{l}_binary_dpp_{o}")
                                    nn_relu_bin[bn][l][o] = cp.Variable((D_S_ts,),
                                                                        name=f"nn_layer_{l}_binary_{o}")
                                    C_bounds_relu_bin += [nn_relu_bin_dpp[bn][l][o] == nn_relu_bin[bn][l][o]]
                                case "qp":
                                    # Binaries are continuous variables between 0 and 1
                                    nn_relu_bin[bn][l][o] = cp.Variable((D_S_ts,), name=f"nn_layer_{l}_binary_{o}")
                                    C_bounds_relu_bin += [0 <= nn_relu_bin[bn][l][o]]
                                    C_bounds_relu_bin += [nn_relu_bin[bn][l][o] <= 1]
                                case _:
                                    raise ValueError(f"The formulation '{relu_relaxation}' "
                                                     f"of the ReLU binaries is not recognized")

                    ### Constraints
                    for ts_idx, ts in enumerate(S_ts):
                        # bound constraints
                        # TODO: Add the bounds on nn_inputs which are parameters
                        C_bounds_nn += [nn_zs[bn][l][z][ts_idx] <= z_bounds[bn][l][z][ts_idx, 1] for z in
                                        S_nn_zs[bn][l]]
                        C_bounds_nn += [nn_zs[bn][l][z][ts_idx] >= z_bounds[bn][l][z][ts_idx, 0] for z in
                                        S_nn_zs[bn][l]]
                        if not isinstance(layer_onnx, InputLayer):  # input layer has no zhat
                            C_bounds_nn += [nn_zhats[bn][l][z][ts_idx] <= zhat_bounds[bn][l][z][ts_idx, 1] for z in
                                            S_nn_zhats[bn][l]]
                            C_bounds_nn += [nn_zhats[bn][l][z][ts_idx] >= zhat_bounds[bn][l][z][ts_idx, 0] for z in
                                            S_nn_zhats[bn][l]]

                        # input constraints: for the input layer only
                        if l == nn_input_layer_idx[bn]:
                            C_input_nn_tin += [nn_inputs[bn][o][ts_idx] == t_in[bn][zn][ts_idx]
                                               for o, zn in enumerate(S_zones[bn])]
                            C_input_nn_phvac += [
                                nn_inputs[bn][len(S_zones[bn]) + o][ts_idx] ==
                                p_heating[bn][zn][ts_idx] - p_cooling[bn][zn][ts_idx]  # * 1000
                                for o, zn in enumerate(S_zones[bn])]
                            C_input_nn_tamb += [nn_inputs[bn][2 * len(S_zones[bn])][ts_idx] == t_amb[ts_idx]]
                            if len(S_nn_inputs[bn]) > 2 * len(S_zones[bn]) + 1:
                                C_input_nn_t += [nn_inputs[bn][2 * len(S_zones[bn]) + 1][ts_idx] == hours[ts_idx]]
                            # input scaling: output of input layer is the input variable scaled
                            tmp_scaled_input = nn_onnx[bn].scaling_object.get_scaled_input_expressions(
                                {i: nn_inputs[bn][i][ts_idx] for i in S_nn_inputs[bn]})
                            C_input_scaling += [nn_zs[bn][l][z][ts_idx] == tmp_scaled_input[z]
                                                for z in S_nn_zs[bn][l]]
                        # Hidden Layer
                        # For all layers except the input layer: see Murzakanov et al. (2022) for the formulation
                        else:
                            # How it works:
                            #   layer l input and output are in terms of the preactivation + activation function
                            #   input of layer l = output of layer l-1
                            #   output of layer l = preactivation + activation function of the input of layer l

                            # linear part: preactivation
                            C_preactivation += [nn_zhats[bn][l][o][ts_idx] == cp.sum(
                                [nn_zs[bn][l - 1][i][ts_idx] * nn_weights[bn][l][i][o] for i in S_nn_zs[bn][l - 1]])
                                                + nn_biases[bn][l][o] for o in S_nn_zhats[bn][l]]
                            # C_test += [nn_zhats[bn][1][o][ts_idx] == self.pm_layer_1_zhat[0][o][ts_idx] for o in S_nn_zhats[bn][1]]
                            if nn_activations[bn][l] == "relu":
                                # binary part
                                C_hidden_layer += [nn_zs[bn][l][z][ts_idx] >= nn_zhats[bn][l][z][ts_idx]
                                                   for z in S_nn_zhats[bn][l]]
                                C_hidden_layer += [nn_zs[bn][l][z][ts_idx] <=
                                                   nn_relu_bin[bn][l][z][ts_idx] * zhat_bounds[bn][l][z][ts_idx, 1]
                                                   for z in S_nn_zhats[bn][l]]
                                C_hidden_layer += [nn_zs[bn][l][z][ts_idx] >= 0 for z in S_nn_zhats[bn][l]]
                                # C_hidden_layer += [nn_zhats_dpp[bn][l][z][ts_idx] == nn_zhats[bn][l][z][ts_idx] - (
                                #         1 - nn_relu_bin[bn][l][z][ts_idx]) for z in S_nn_zhats[bn][l]]
                                C_hidden_layer += [nn_zs[bn][l][z][ts_idx] <= nn_zhats[bn][l][z][ts_idx] - (
                                        1 - nn_relu_bin[bn][l][z][ts_idx]) * zhat_bounds[bn][l][z][ts_idx, 0] for z in
                                                   S_nn_zhats[bn][l]]
                            elif nn_activations[bn][l] == "leakyrelu":
                                alpha = 0.01
                                # binary part
                                C_hidden_layer += [nn_zs[bn][l][z][ts_idx] >= nn_zhats[bn][l][z][ts_idx]
                                                   for z in S_nn_zhats[bn][l]]
                                C_hidden_layer += [nn_zs[bn][l][z][ts_idx] <=
                                                   nn_relu_bin[bn][l][z][ts_idx] * zhat_bounds[bn][l][z][ts_idx, 1]
                                                   + alpha * nn_zhats[bn][l][z][ts_idx]
                                                   for z in S_nn_zhats[bn][l]]
                                C_hidden_layer += [nn_zs[bn][l][z][ts_idx] >= alpha * nn_zhats[bn][l][z][ts_idx]
                                                   for z in S_nn_zhats[bn][l]]
                                # C_hidden_layer += [nn_zhats_dpp[bn][l][z][ts_idx] == nn_zhats[bn][l][z][ts_idx] - (
                                #         1 - nn_relu_bin[bn][l][z][ts_idx]) for z in S_nn_zhats[bn][l]]
                                C_hidden_layer += [nn_zs[bn][l][z][ts_idx] <= nn_zhats[bn][l][z][ts_idx] - (
                                        1 - nn_relu_bin[bn][l][z][ts_idx]) * (1 + alpha) * zhat_bounds[bn][l][z][ts_idx, 0]
                                                   for z in S_nn_zhats[bn][l]]
                            else:
                                C_hidden_layer += [nn_zs[bn][l][z][ts_idx] == nn_zhats[bn][l][z][ts_idx] for z in
                                                   S_nn_zhats[bn][l]]
                        # output constraints: for the output layer only
                        if l == S_nn_layers[bn][-1]:
                            tmp_unscaled_output = nn_onnx[bn].scaling_object.get_unscaled_output_expressions(
                                {o: nn_zs[bn][l][o][ts_idx] for o in S_nn_zs[bn][l]})
                            C_output_unscale += [nn_unscaled_outputs[bn][o][ts_idx] == tmp_unscaled_output[o]
                                                 for o in S_nn_zs[bn][l]]
                            if target_name == "Zone Mean Air Temperature(t+1)":
                                C_output_nn += [nn_unscaled_outputs[bn][o][ts_idx] == t_in[bn][zn][ts_idx + 1]
                                                for o, zn in enumerate(S_zones[bn])]
                            elif target_name == "Delta Mean Air Temperature(t)":  # Delta T
                                C_output_nn += [
                                    nn_unscaled_outputs[bn][o][ts_idx] + t_in[bn][zn][ts_idx] == t_in[bn][zn][
                                        ts_idx + 1]
                                    for o, zn in enumerate(S_zones[bn])]

                return {
                    "Constraints": [C_input_nn_tin, C_input_nn_phvac, C_input_nn_tamb, C_input_nn_t, C_input_scaling,
                                    C_preactivation, C_output_nn, C_output_unscale, C_bounds_nn, C_bounds_relu_bin,
                                    C_hidden_layer, C_additional],
                    "Sets": [S_nn_layers, S_nn_zs],
                    "Parameters": [nn_weights, nn_biases],
                    "Variables": [nn_inputs, nn_zhats, nn_zs, nn_relu_bin],
                    "Infos": [nn_activations, nn_is_input_layer, nn_input_layer_idx]}

        def rc_formulation():
            # new parameters
            rc_alpha = {bn: {zn: {zzn: cp.Parameter(name=f"rc_alpha_{zn}_x_{zzn}")
                                  for zzn in S_zones[bn]}
                             for zn in S_zones[bn]}
                        for bn in S_buildings}
            rc_inv_R = {bn: {zn: cp.Parameter(name=f"rc_inv_R_{zn}") for zn in S_zones[bn]}
                        for bn in S_buildings}
            rc_inv_C = {bn: {zn: cp.Parameter(name=f"rc_inv_C_{zn}") for zn in S_zones[bn]}
                        for bn in S_buildings}
            rc_h_eff = {bn: {zn: cp.Parameter(name=f"rc_h_eff_{zn}") for zn in S_zones[bn]}
                        for bn in S_buildings}
            rc_c_eff = {bn: {zn: cp.Parameter(name=f"rc_c_eff_{zn}") for zn in S_zones[bn]}
                        for bn in S_buildings}
            # intermediary variables to keep the problem DPP
            q_heating = {bn: {zn: cp.Variable((D_S_ts,), name=f"rc_q_heating_{zn}") for zn in S_zones[bn]}
                         for bn in S_buildings}
            q_cooling = {bn: {zn: cp.Variable((D_S_ts,), name=f"rc_q_cooling_{zn}") for zn in S_zones[bn]}
                         for bn in S_buildings}
            t_amb_div_R_dpp = {bn: {zn: cp.Variable((D_S_ts,), name=f"rc_t_amb_div_R_{zn}") for zn in S_zones[bn]}
                               for bn in S_buildings}
            t_in_div_R_dpp = {bn: {zn: cp.Variable((D_S_ts,), name=f"rc_t_in_div_R_{zn}") for zn in S_zones[bn]}
                              for bn in S_buildings}
            t_amb_var_dpp = cp.Variable((D_S_ts,), name=f"rc_t_amb_var_dpp")
            # Constraints for the RC model
            C_rc = []
            for bn in S_buildings:
                for t_idx, t in enumerate(S_ts):
                    for z, zn in enumerate(S_zones[bn]):
                        # intermediary power
                        C_rc.append(p_hvac[bn][zn][t_idx] == p_cooling[bn][zn][t_idx] + p_heating[bn][zn][t_idx])
                        ### To keep problem DPP
                        # q heating
                        C_rc.append(q_heating[bn][zn][t_idx] == rc_h_eff[bn][zn] * p_heating[bn][zn][t_idx])
                        # q cooling
                        C_rc.append(q_cooling[bn][zn][t_idx] == rc_c_eff[bn][zn] * p_cooling[bn][zn][t_idx])
                        # intermediary t_amb variables (which is obviously equal to the parameter)
                        C_rc.append(t_amb_var_dpp[t_idx] == t_amb[t_idx])
                        # intermediary variable: t_amb * 1/R (var * param)
                        C_rc.append(t_amb_div_R_dpp[bn][zn][t_idx] == t_amb_var_dpp[t_idx] * rc_inv_R[bn][zn])
                        # intermediary variable: t_in * 1/R (var * param)
                        C_rc.append(t_in_div_R_dpp[bn][zn][t_idx] == t_in[bn][zn][t_idx] * rc_inv_R[bn][zn])

                        ### RC model
                        tmp0 = sum(t_in[bn][zzn][t_idx] * rc_alpha[bn][zn][zzn] for zzn in S_zones[bn])
                        tmp1 = q_heating[bn][zn][t_idx] * rc_inv_C[bn][zn]
                        tmp2 = - q_cooling[bn][zn][t_idx] * rc_inv_C[bn][zn]
                        tmp3 = t_amb_div_R_dpp[bn][zn][t_idx] * rc_inv_C[bn][zn]
                        tmp4 = - t_in_div_R_dpp[bn][zn][t_idx] * rc_inv_C[bn][zn]
                        if target_name == "Zone Mean Air Temperature(t+1)":
                            C_rc.append(t_in[bn][zn][t_idx + 1] == tmp0 + tmp1 + tmp2 + tmp3 + tmp4)
                        elif target_name == "Delta Mean Air Temperature(t)":  # Delta T
                            C_rc.append(
                                t_in[bn][zn][t_idx + 1] == tmp0 + tmp1 + tmp2 + tmp3 + tmp4 + t_in[bn][zn][t_idx])

            return {"Parameters": [rc_alpha, rc_h_eff, rc_c_eff, rc_inv_R, rc_inv_C],
                    "Variables": [p_cooling, p_heating],
                    "Constraints": [C_rc]}

        ####################
        #  Sets
        ####################
        # Sets common to the community
        S_ts = self.ts_ems[:-1]  # time steps (midnight not included)
        S_ts_mn = self.ts_ems  # time steps (midnight included)
        S_buildings = self.bldg_assets.index.to_list()  # set of building names
        # Sets for each building
        S_zones = dict()  # keys = building names, values = list of zone names
        S_acus = dict()  # keys = building names, values = list of ACU names
        for bn in S_buildings:  # bn is the building name
            # Zone and ACU names for the building bn
            S_zones[bn] = self.bldg_assets[bn].zones_df_no_plenum["name"].to_list()
            S_acus[bn] = self.bldg_assets[bn].ACUs["ACU"].to_list()

        # zones served by an ACU
        S_acu_zones = dict()
        for bn in S_buildings:
            # groupby acu name, make a list of the zone name in each group,
            # transform the tuple (acu_name, list of zone names) to a dict
            zonesbyacu = self.bldg_assets[bn].zones_df_no_plenum.groupby("ACU")["name"].apply(list).to_dict()
            S_acu_zones[bn] = zonesbyacu

        # dimensions of the sets
        D_S_ts = len(S_ts)
        D_S_ts_mn = len(S_ts_mn)
        D_S_buildings = len(S_buildings)
        D_S_zones = {bn: len(S_zones[bn]) for bn in S_buildings}
        D_S_acus = {bn: len(S_acus[bn]) for bn in S_buildings}

        ####################
        #  Parameters
        ####################
        # exogenous variables
        t_amb = cp.Parameter((D_S_ts,), name=f"t_amb")
        hours = cp.Parameter((D_S_ts,), name=f"hours")
        nd_load = {bn: cp.Parameter((D_S_ts,), name=f"nd_load") for bn in S_buildings}

        # constant parameters
        line_capacity = cp.Parameter((1,), name="line_capacity")
        demand_charge = cp.Parameter((1,), name="demand_charge")
        prices_import = cp.Parameter((D_S_ts,), name="prices_import")
        prices_export = cp.Parameter((D_S_ts,), name="prices_export")
        ACU_capacities = {bn: {acu: cp.Parameter((1,), name=f"ACU_capacity_{acu}") for acu in S_acus[bn]}
                          for bn in S_buildings}
        hvac_capacities = {bn: {zn: cp.Parameter((1,), name=f"hvac_capacity_{zn}") for zn in S_zones[bn]}
                           for bn in S_buildings}
        Tin0 = {bn: {zn: cp.Parameter((1,), name=f"Tin0_{zn}") for zn in S_zones[bn]} for bn in S_buildings}
        Tmin = {bn: {zn: cp.Parameter((D_S_ts_mn,), name=f"Tmin_{zn}") for zn in S_zones[bn]} for bn in S_buildings}
        Tmax = {bn: {zn: cp.Parameter((D_S_ts_mn,), name=f"Tmax_{zn}") for zn in S_zones[bn]} for bn in S_buildings}

        ####################
        #  Variables
        ####################
        # Community-level variables
        p_demand_max = cp.Variable(name="p_demand_max", nonneg=True)
        p_import = cp.Variable((D_S_ts,), name="p_import", nonneg=True)  # no import decision for midnight next day
        p_export = cp.Variable((D_S_ts,), name="p_export", nonneg=True)  # no export decision for midnight next day
        # Buidling-level variables
        # -> cf. constraints
        # Zone-level variables
        # /!\ p_hvac is a dictionary of dictionary of cp.variables, it is not a cp.variable itself
        p_hvac = {bn: {zn: cp.Variable((D_S_ts,), name=f"p_hvac_{zn}", nonneg=True) for zn in S_zones[bn]} for bn in
                  S_buildings}
        # variable for cooling and heating power (must be positive)
        p_cooling = {bn: {zn: cp.Variable((D_S_ts,), name=f"p_cooling_{zn}", nonneg=True) for zn in S_zones[bn]}
                     for bn in S_buildings}
        p_heating = {bn: {zn: cp.Variable((D_S_ts,), name=f"p_heating_{zn}", nonneg=True) for zn in S_zones[bn]}
                     for bn in S_buildings}
        t_in = {bn: {zn: cp.Variable((D_S_ts_mn,), name=f"t_in_{zn}") for zn in S_zones[bn]} for bn in S_buildings}

        ####################
        #  Constraints
        ####################
        ### Community-level constraints
        # the community can not import more than the line capacity
        C_line_capacity = [p_demand_max <= line_capacity]
        # the peak demand is the maximum of the import minus the export (constraint for each ts)
        C_peak_demand = [p_demand_max >= p_import[ts_idx] - p_export[ts_idx] for ts_idx, _ in enumerate(S_ts)]
        # the power balance (constraint for each ts, sum over all buildings)
        C_power_balance = [p_import[ts_idx] - p_export[ts_idx] == sum(nd_load[bn][ts_idx] +
                                                                      sum(p_hvac[bn][zn][ts_idx] for zn in S_zones[bn])
                                                                      for bn in S_buildings)
                           for ts_idx, ts in enumerate(S_ts)]

        ### Building-level constraints
        # build the nn formulation
        if thermal_model == "nn":
            nn_formltn = build_nn_formulation(relu_relaxation)
            S_nn_layers, S_nn_zs = nn_formltn["Sets"]
            nn_inputs, nn_zhats, nn_zs, nn_relu_bin = nn_formltn["Variables"]
            nn_weights, nn_biases = nn_formltn["Parameters"]
            C_thermal_model = [c for c_l in nn_formltn["Constraints"] for c in c_l]
            nn_infos = nn_formltn["Infos"]
            nn_activations = nn_infos[0]
        # build the RC formulation
        elif thermal_model == "rcmodel" or thermal_model == "spatialrcmodel" :
            C_thermal_model = [c for c_l in rc_formulation()["Constraints"] for c in c_l]
        else:
            raise ValueError(f"Unknown thermal model {thermal_model}. Possibilities are: '(spatial)rcmodel' or 'nn'")

        ### ACU level constraints
        C_acu_capacity = []
        for bn in S_buildings:
            for acu in S_acus[bn]:
                C_acu_capacity.extend([sum(p_hvac[bn][zn][ts_idx] for zn in S_acu_zones[bn][acu]) <=
                                       ACU_capacities[bn][acu] for ts_idx, _ in enumerate(S_ts)])

        ### Zone level constraints
        C_init_temp = []
        C_cooling_setpoint = []
        C_heating_setpoint = []
        C_phvac = []
        C_hvac_capacity = []
        C_dummymodel = []
        for bn in S_buildings:
            for zn in S_zones[bn]:
                C_init_temp.append(t_in[bn][zn][0] == Tin0[bn][zn])
                C_cooling_setpoint.extend([t_in[bn][zn][0] <= Tin0[bn][zn]] +
                                          [t_in[bn][zn][ts_idx] <= Tmax[bn][zn][ts_idx]
                                           for ts_idx, ts in enumerate(S_ts_mn)][1:])
                C_heating_setpoint.extend([t_in[bn][zn][0] >= Tin0[bn][zn]] +
                                          [t_in[bn][zn][ts_idx] >= Tmin[bn][zn][ts_idx]
                                           for ts_idx, ts in enumerate(S_ts_mn)][1:])
                C_phvac.extend([p_hvac[bn][zn][ts_idx] == p_cooling[bn][zn][ts_idx] + p_heating[bn][zn][ts_idx]
                                    for ts_idx, _ in enumerate(S_ts)])
                C_hvac_capacity.extend([p_cooling[bn][zn][ts_idx] <= 0.8 * hvac_capacities[bn][zn]
                                        for ts_idx, _ in enumerate(S_ts)])
                C_hvac_capacity.extend([p_hvac[bn][zn][ts_idx] <= hvac_capacities[bn][zn]
                                        for ts_idx, _ in enumerate(S_ts)])
                C_dummymodel.extend([t_in[bn][zn][ts_idx + 1] == 0.8 * t_in[bn][zn][ts_idx] + 4 * p_hvac[bn][zn][ts_idx]
                                     for ts_idx, _ in enumerate(S_ts)])

        constraints = (C_line_capacity + C_peak_demand + C_power_balance + C_acu_capacity + C_init_temp + C_phvac
                       # + C_cooling_setpoint + C_heating_setpoint
                       + C_hvac_capacity + C_thermal_model)  # + C_dummymodel

        ####################
        #  Objective
        ####################
        # The occupancy weights must be initialized in the problem otherwise it is not identified as
        # Disciplined Convex Programming (DCP) since the square of the temperature difference is convex
        # but the weight could be negative.
        occupancy_weights = {bn: {zn: self.bldg_assets[bn].zone_assets[zn].occupancy_weights.values
                                  for zn in S_zones[bn]} for bn in S_buildings}

        # consigne de temperature: Ideally 21° constantly
        temperature_rule = {
            bn: {zn: cp.Parameter((D_S_ts_mn,), name=f"temp_rule_{zn}")  # [21 for ts_idx, _ in enumerate(S_ts_mn)]
                 for zn in S_zones[bn]} for bn in S_buildings}
        # the objective function
        # (Un)comment the last two lines to make the ridge regression and thus, the problem convex
        obj = cp.Minimize(demand_charge * p_demand_max + sum(
            prices_import[ts_idx] * p_import[ts_idx] - prices_export[ts_idx] * p_export[ts_idx]
            for ts_idx, _ in enumerate(S_ts)) * self.dt_ems
                          # decision variable regularization
                          # + 1e-3 * sum(p_import[ts_idx] ** 2 + p_export[ts_idx] ** 2  # for ridge regression (1)
                          #              for ts_idx, ts in enumerate(S_ts))  # for ridge regression (2)
                          # temperature regularization
                          + sum(occupancy_weights[bn][zn][ts_idx] * (
                                                        t_in[bn][zn][ts_idx] - temperature_rule[bn][zn][ts_idx]) ** 2
                                for bn in S_buildings
                                for zn in S_zones[bn] for ts_idx, _ in enumerate(S_ts_mn))
                          )

        ####################
        #  Solution
        ####################
        prob = cp.Problem(obj, constraints)
        print(f"Problem is DPP: {prob.is_dpp()}")

        return prob

    def set_zhat_and_z_bounds(self):
        """
        Compute the bounds of zhat. It is assumed there is a z variable in the previous layer.
        Activation function is assumed to be ReLU.
        The first layer is assumed to be the input layer.
        :param prob: the cvxpy problem WITH THE WEIGHTS AND BIASES ALREADY DEFINED
        :return:
        """

        prob = self.cvxpy_opti_formulation
        def scale_one_input(input, input_idx, scaler):
            """
            Scale one input
            :param input: the input to scale
            :param scaler: the scaler to use
            :return: the scaled input
            """
            filler = np.zeros((len(input), len(scaler.feature_names_in_)))
            filler[:, input_idx] = input
            # suppress the warning of the scaler because input has no name
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scaled_input = np.array(
                    [scaler.transform(s.reshape(1, -1))[0, input_idx] for s in filler])
            return scaled_input

        for b in self.bldg_assets:
            # read the nn from onnx
            layers_onnx = list(b.nn.layers)
            bn = b.name
            # set the value of the bounds
            for l, layer_onnx in enumerate(layers_onnx):
                for o in [i[0] for i in layer_onnx.output_indexes]:
                    if isinstance(layer_onnx, InputLayer):
                        # Set the parameter input bounds to the value of the parameter
                        # /!\ the parameter must be scaled
                        if (b.nn_scaler['input'].feature_names_in_[o]
                                == "Site Outdoor Air Drybulb Temperature,ENVIRONMENT"):
                            t_amb = prob.param_dict.get(f"t_amb").value  # vertical vector
                            t_amb_scaled = scale_one_input(t_amb, o, b.nn_scaler['input'])
                            t_amb_scaled_bounds = np.stack([t_amb_scaled, t_amb_scaled], -1)
                            prob.param_dict.get(f"nn_layer_{l}_z_bound_{o}").value = t_amb_scaled_bounds
                        # for the hour
                        elif b.nn_scaler['input'].feature_names_in_[o] == "Hour":
                            hours = prob.param_dict.get(f"hours").value
                            hours_scaled = scale_one_input(hours, o, b.nn_scaler['input'])
                            hours_scaled_bounds = np.stack([hours_scaled, hours_scaled], -1)
                            prob.param_dict.get(f"nn_layer_{l}_z_bound_{o}").value = hours_scaled_bounds
                        else:
                            # Set the bounds of the z variables for the input layer. These are the scaled input bound.
                            # They are constant over time because the input variable should be allowed to take any
                            # value within the bounds.
                            prob.param_dict.get(f"nn_layer_{l}_z_bound_{o}").value = np.array(
                                [b.nn_scaled_input_bounds[o] for t in self.ts_ems[:-1]])
                    else:
                        # Set the bounds of the zhat variables
                        # Zhat_ub is w * z_ub is w is positive, w * z_lb if w is negative
                        lb = 0
                        ub = 0
                        for i in range(nb_neuron_previous_layer):
                            prv_z_lb = prob.param_dict.get(f"nn_layer_{l - 1}_z_bound_{i}").value[:, 0]
                            prv_z_ub = prob.param_dict.get(f"nn_layer_{l - 1}_z_bound_{i}").value[:, 1]
                            w = prob.param_dict.get(f"nn_layer_{l}_weights").value[i, o]
                            lb += prv_z_lb * w if w > 0 else prv_z_ub * w
                            ub += prv_z_ub * w if w > 0 else prv_z_lb * w
                        lb += prob.param_dict.get(f"nn_layer_{l}_biases").value[o]
                        ub += prob.param_dict.get(f"nn_layer_{l}_biases").value[o]
                        prob.param_dict.get(f"nn_layer_{l}_zhat_bound_{o}").value = np.stack([lb, ub], axis=-1)

                        # Set the bounds of the z variables
                        if layer_onnx.activation == "relu":
                            # Apply the relu function to the bounds
                            prob.param_dict.get(f"nn_layer_{l}_z_bound_{o}").value = np.stack(
                                [np.max([prob.param_dict.get(f"nn_layer_{l}_zhat_bound_{o}").value[:, 0],
                                         np.zeros(prob.param_dict.get(f"nn_layer_{l}_zhat_bound_{o}").value[:, 0].shape)
                                         ], axis=0),
                                 np.max([prob.param_dict.get(f"nn_layer_{l}_zhat_bound_{o}").value[:, 1],
                                         np.zeros(prob.param_dict.get(f"nn_layer_{l}_zhat_bound_{o}").value[:, 1].shape)
                                         ], axis=0)],
                                axis=-1)
                        elif layer_onnx.activation == "leakyrelu":
                            # Apply the leaky relu function to the bounds
                            alpha = 0.01
                            prob.param_dict.get(f"nn_layer_{l}_z_bound_{o}").value = np.stack(
                                [np.max([prob.param_dict.get(f"nn_layer_{l}_zhat_bound_{o}").value[:, 0],
                                     alpha * prob.param_dict.get(f"nn_layer_{l}_zhat_bound_{o}").value[:, 0]], axis=0),
                                 np.max([prob.param_dict.get(f"nn_layer_{l}_zhat_bound_{o}").value[:, 1],
                                     alpha * prob.param_dict.get(f"nn_layer_{l}_zhat_bound_{o}").value[:, 1]], axis=0)],
                                axis=-1)
                        elif layer_onnx.activation == "linear":  # the ouput layer has a linear activation
                            prob.param_dict.get(f"nn_layer_{l}_z_bound_{o}").value = prob.param_dict.get(
                                f"nn_layer_{l}_zhat_bound_{o}").value
                        else:
                            raise ValueError(f"Activation function '{layer_onnx.activation}' not recognized")
                nb_neuron_previous_layer = o + 1

    def set_cvxpy_parameters(self, weights=None, biases=None, relu_binaries=None, nd_load_flag=True):
        """
        Take all the information given to the ems object before (market, buildings, nn, etc.) and set the cvxpy
        parameters in the cvxpy problem based on these info.
        Args:
            prob:
            weights:
            biases:
            relu_binaries:

        Returns:

        """

        self.nd_load_flag = nd_load_flag
        prob = self.cvxpy_opti_formulation

        def update_setpoints(S_ts_mn):
            """
            Update the setpoints of the zones
            :param heating_setpoint: the heating setpoint Series (midnight included)
            :param cooling_setpoint: the cooling setpoint Series (midnight included)
            :return:
            """
            for b in self.bldg_assets:
                for zn in b.zones_df_no_plenum["name"]:
                    Tmin0 = b.zone_assets[zn].Tmin.loc[S_ts_mn[0]]
                    Tmax0 = b.zone_assets[zn].Tmax.loc[S_ts_mn[0]]
                    # bar and restaurants at the bottom
                    if "bot" in zn:
                        # 16° until 8am, 18° until 11am, 21° until midnight
                        heating_setpoint = np.array([Tmin0] + [16] * 8 + [18] * 3 + [21] * 13)
                        # 24° until 8am, after 23° until midnight
                        cooling_setpoint = np.array([Tmax0] + [24] * 8 + [23] * 3 + [23] * 13)
                    # constrain the temperature in the middle a bit more
                    elif "mid" in zn:
                        heating_setpoint = np.concatenate((Tmin0.reshape(1, ), np.clip(
                            b.zone_assets[zn].Tmin.loc[S_ts_mn[1:]].values, 17, 21)), axis=0)
                        cooling_setpoint = np.concatenate((Tmin0.reshape(1, ), np.clip(
                            b.zone_assets[zn].Tmax.loc[S_ts_mn[1:]].values - 1, 21, 24)), axis=0)
                    # housing in the top floor
                    elif "top" in zn:
                        # 17° until 6am, 21° until 8am, 16° until 5pm, 21° until 10pm, 17° until midnight
                        heating_setpoint = np.array([Tmin0] + [17] * 6 + [21] * 2 + [16] * 9 + [21] * 5 + [17] * 2)
                        # 23° until 8am, 26° until 5pm, 23° until midnight
                        cooling_setpoint = np.array([Tmax0] + [23] * 8 + [26] * 9 + [23] * 7)
                    else:
                        raise ValueError(f"Zone '{zn}' not recognized")

                    # b.zone_assets[zn].Tmin.loc[S_ts_mn] = heating_setpoint
                    # b.zone_assets[zn].Tmax.loc[S_ts_mn] = cooling_setpoint
                    b.zone_assets[zn].Tmin.loc[S_ts_mn[1:5]] = b.zone_assets[zn].Tmin.loc[S_ts_mn[1:5]] - 0.5
                    b.zone_assets[zn].Tmin.loc[S_ts_mn[5:]] = b.zone_assets[zn].Tmin.loc[S_ts_mn[5:]] + 0.2
                    b.zone_assets[zn].Tmax.loc[S_ts_mn[1:5]] = b.zone_assets[zn].Tmax.loc[S_ts_mn[1:5]] + 0.5
                    b.zone_assets[zn].Tmax.loc[S_ts_mn[5:]] = b.zone_assets[zn].Tmax.loc[S_ts_mn[5:]] - 0.2

        # Recreate the sets
        S_ts = self.ts_ems[:-1]  # time steps (midnight not included)
        S_ts_mn = self.ts_ems  # time steps (midnight included)
        S_buildings = self.bldg_assets.index.to_list()  # set of building names
        S_zones = {bn: self.bldg_assets[bn].zones_df_no_plenum["name"].to_list() for bn in S_buildings}
        S_ACUs = {bn: self.bldg_assets[bn].ACUs["ACU"] for bn in S_buildings}
        D_S_ts = len(S_ts)

        # Extract the parameter references from the cvxpy problem
        line_capacity = prob.param_dict.get("line_capacity")
        demand_charge = prob.param_dict.get("demand_charge")
        prices_import = prob.param_dict.get("prices_import")
        prices_export = prob.param_dict.get("prices_export")
        t_amb = prob.param_dict.get("t_amb")
        hours = prob.param_dict.get("hours")
        nd_load = {bn: prob.param_dict.get(f"nd_load") for bn in S_buildings}
        ACU_capacities = {bn: {acu: prob.param_dict.get(f"ACU_capacity_{acu}") for acu in S_ACUs[bn]}
                          for bn in S_buildings}
        hvac_capacities = {bn: {zn: prob.param_dict.get(f"hvac_capacity_{zn}") for zn in S_zones[bn]}
                           for bn in S_buildings}
        Tin0 = {bn: {zn: prob.param_dict.get(f"Tin0_{zn}") for zn in S_zones[bn]}
                for bn in S_buildings}
        Tmin = {bn: {zn: prob.param_dict.get(f"Tmin_{zn}") for zn in S_zones[bn]}
                for bn in S_buildings}
        Tmax = {bn: {zn: prob.param_dict.get(f"Tmax_{zn}") for zn in S_zones[bn]}
                for bn in S_buildings}
        temperature_rule = {bn: {zn: prob.param_dict.get(f"temp_rule_{zn}") for zn in S_zones[bn]}
                            for bn in S_buildings}

        # update the setpoint values
        # update_setpoints(S_ts_mn)

        # set the values of the parameters only if not performing the cvxpylayer formulation
        line_capacity.value = np.array(self.market.line_capacity).reshape(1, )
        demand_charge.value = np.array(self.market.demand_charge).reshape(1, )
        prices_import.value = self.market.prices_import.loc[S_ts].values
        prices_export.value = self.market.prices_export.loc[S_ts].values
        bn = ""
        for b in self.bldg_assets:
            bn = b.name
            # To better see the scheduling, the base load is assumed to be 0 at the moment
            if self.nd_load_flag:
                nd_load[bn].value = self.bldg_assets[bn].nd_load.loc[S_ts].values
            else:
                nd_load[bn].value = np.zeros(D_S_ts)
            for acu in S_ACUs[bn]:
                ACU_capacities[bn][acu].value = np.array(self.bldg_assets[bn].ACU_capacity[acu]).reshape(1, )
            for zn in S_zones[bn]:
                hvac_capacities[bn][zn].value = np.array(
                    self.bldg_assets[bn].zone_assets[zn].hvac_capacity).reshape(1, )
                Tin0[bn][zn].value = np.array(self.bldg_assets[bn].zone_assets[zn].Tin0).reshape(1, )
                # Tmin and Tmax might be none because of the temperature regularization (rather than constraint)
                if Tmin[bn][zn] is not None:
                    Tmin[bn][zn].value = self.bldg_assets[bn].zone_assets[zn].Tmin.loc[S_ts_mn].values
                if Tmax[bn][zn] is not None:
                    Tmax[bn][zn].value = self.bldg_assets[bn].zone_assets[zn].Tmax.loc[S_ts_mn].values
                # obj function parameters
                if temperature_rule[bn][zn] is not None:
                    temperature_rule[bn][zn].value = self.bldg_assets[bn].zone_assets[zn].Ttgt.loc[S_ts_mn].values
        # same ambient temperature for all buildings
        if t_amb is not None:
            t_amb.value = self.bldg_assets[bn].simulation.loc[
                S_ts, "Site Outdoor Air Drybulb Temperature,ENVIRONMENT"].values
        if hours is not None:
            hours.value = self.bldg_assets[bn].simulation.loc[S_ts, "Hour"].values

        if self.thermal_model == "nn":
            # set the NN parameters
            for b in self.bldg_assets:
                # read the nn from onnx
                layers_onnx = list(b.nn.layers)
                bn = b.name
                # Set the values of the NN parameters
                for l, layer_onnx in enumerate(layers_onnx):
                    if not isinstance(layer_onnx, InputLayer):
                        prob.param_dict.get(f"nn_layer_{l}_weights").value = weights[bn][l].detach().cpu().numpy()
                        prob.param_dict.get(f"nn_layer_{l}_biases").value = biases[bn][l].detach().cpu().numpy()
                    # set the value of the binaries (if they are to be fixed)
                    if layer_onnx.activation == "relu" and relu_binaries is not None:
                        for o in [i[0] for i in layer_onnx.output_indexes]:
                            prob.param_dict.get(f"nn_layer_{l}_binary_dpp_{o}").value = relu_binaries[bn][l][o]
            # set the value of the bounds
            self.set_zhat_and_z_bounds()
        # for the rcmodel
        elif self.thermal_model == "rcmodel" or self.thermal_model == "spatialrcmodel":
            for bn in S_buildings:
                for zn in S_zones[bn]:
                    # WARNING: the numpy array and the original tensor share the same memory
                    # check numpy in torch doc for more information
                    # try:
                    #     print(f"alpha keys: {list(weights[bn][zn]['alpha'].keys())}")  # DEBUG
                    # except KeyError:
                    #     print(f"weights[bn]: {weights[bn]}")
                    param_dict = prob.param_dict
                    for zzn in S_zones[bn]:
                        if self.thermal_model == "spatialrcmodel":  # if it is a spatial rcmodel
                            alpha = weights[bn][zn]["alpha"][zzn].detach().numpy()
                        else:  # if it is a classical "rcmodel"
                            alpha = 1 if zn == zzn else 0
                        param_dict.get(f"rc_alpha_{zn}_x_{zzn}").value = alpha
                        v = alpha
                        if np.isnan(v).any():
                            print(f"variable is nan: {v}")
                        elif np.isinf(v).any():
                            print(f"variable is inf: {v}")
                        elif 0 < v < 1e-5:
                            print(f"variable is too small: {v}")
                        elif v > 1e5:
                            print(f"variable is too large: {v}")
                    h_eff = weights[bn][zn]["h_eff"].detach().numpy()
                    c_eff = weights[bn][zn]["c_eff"].detach().numpy()
                    R = weights[bn][zn]["R"].detach().item()
                    C = weights[bn][zn]["C"].detach().item()
                    beta_h = h_eff / C
                    beta_c = c_eff / C
                    gamma = 1 / (R * C)
                    for v in [h_eff, c_eff, R, C]:
                        if np.isnan(v).any():
                            print(f"variable is nan: {v}")
                        elif np.isinf(v).any():
                            print(f"variable is inf: {v}")
                        elif 0 < v < 1e-5:
                            print(f"variable is too small: {v}")
                        elif v > 1e5:
                            print(f"variable is too large: {v}")
                    param_dict.get(f"rc_inv_R_{zn}").value = 1 / R
                    param_dict.get(f"rc_inv_C_{zn}").value = 1 / C
                    param_dict.get(f"rc_h_eff_{zn}").value = h_eff
                    param_dict.get(f"rc_c_eff_{zn}").value = c_eff
        else:
            raise ValueError(f"Unknown thermal model {self.thermal_model}. Possibilities are: "
                             "'(spatial)rcmodel' or 'nn'")


    def solve_cvxpy_gurobi(self, solver_parameters):
        # print(cp.installed_solvers())  # to check the installed solvers
        # # Print the objective
        # print(prob.objective)
        # # Print the constraints
        # for constraint in prob.constraints:
        #     print(constraint)

        prob = self.cvxpy_opti_formulation
        # solve the problem
        solver_engine = cp.GUROBI
        try:
            prob.solve(solver=solver_engine, **solver_parameters)
        # if gurobi returns an error (= iff the problem is infeasible), try mosek
        except cp.error.SolverError:
            print("Gurobi returned an error!")
            # for constraint in prob.constraints:
            #     print(constraint)
            # solver_engine = cp.MOSEK
            # solver_parameters = {'verbose': False, "mosek_params": {'MSK_DPAR_OPTIMIZER_MAX_TIME': 60,
            #                                                        'MSK_DPAR_MIO_TOL_REL_GAP': 0.01,
            #                                                        'MSK_IPAR_NUM_THREADS': 10}}
            # prob.solve(solver=solver_engine, **solver_parameters)

        print("status:", prob.status)
        ### analyze the solution and save the results
        # if the solver returned an inaccurate, solve again with a new solver
        # if prob.status == "optimal_inaccurate":
        #     # solver_engine = cp.MOSEK
        #     # solver_parameters = {'verbose': False, "mosek_params": {'MSK_DPAR_OPTIMIZER_MAX_TIME': 60,
        #     #                                                         'MSK_DPAR_MIO_TOL_REL_GAP': 0.01,
        #     #                                                         'MSK_IPAR_NUM_THREADS': 10}}
        #     # prob.solve(solver=solver_engine, **solver_parameters)
        #     solver_engine = cp.GUROBI
        #     solver_parameters = {'verbose': True, 'TimeLimit': 120, 'MIPGap': 0.01, 'Threads': 0, 'WLSTokenRefresh': 0}
        #     prob.solve(reoptimize=True, solver=solver_engine, **solver_parameters)
        #     print("status:", prob.status)
        # if no incumbent has been found within the time limit, multiply the time limit by 10 and resolve.
        while prob.status == "infeasible_inaccurate" and solver_parameters["TimeLimit"] <= 6000:
            new_time_limit = solver_parameters['TimeLimit'] * 10
            new_mip_gap = 0.1
            print(f"Problem not solved after {solver_parameters['TimeLimit']}s. Time limit increased to "
                  f"{new_time_limit}s.")
            solver_parameters["TimeLimit"] = new_time_limit
            solver_parameters["MIPGap"] = new_mip_gap
            # solver_parameters["MIPGap"] *= 10 if solver_parameters["MIPGap"] < 1 else 1
            prob.solve(reoptimize=True, **solver_parameters)
            print("status:", prob.status)
        # if there is a solution
        solcount = prob.solver_stats.extra_stats.SolCount
        print(f'Solution count = {solcount}')

        if solcount >= 1:
            self.feasible = True
            obj_val = prob.value
            try:
                mip_gap = prob.solver_stats.extra_stats.MIPGap
            except AttributeError:
                mip_gap = None
            solving_time = prob.solver_stats.solve_time

            # if the solution is optimal
            if prob.status == "optimal":
                print('* OPTIMISATION SUCCESSFUL *')
                print(f'Solving time = {solving_time:.2f} s')
                print("Solution value:\t", obj_val)
            # if the solution is suboptimal (or not guaranteed), stop due to time budget reached
            elif prob.status == "user_limit":
                print('* OPTIMISATION MET TERMINATION CRITERIA *')
                if mip_gap is not None:
                    print(f'MIP gap = {mip_gap:.2%}')
                print("Solution value:\t", obj_val)

            p_import_val = prob.var_dict["p_import"].value.reshape(-1, 1)
            p_export_val = prob.var_dict["p_export"].value.reshape(-1, 1)
            if ((p_import_val * p_export_val).round(3) != 0).any():
                print(f"p_import_val: {p_import_val}")
                print(f"p_export_val: {p_export_val}")
                raise ValueError('Warning: Import and export powers cannot be non-zero at the same time')

            p_demand_max_val = prob.var_dict["p_demand_max"].value
            p_demand_max_vec = np.full((self.nb_ts_ems - 1, 1), p_demand_max_val)

            # store the results
            for b in self.bldg_assets:
                bn = b.name
                variables = []
                for z in b.zone_assets:
                    if z.controlled:
                        zn = z.name
                        t_in_val = prob.var_dict[f"t_in_{zn}"].value.reshape(-1, 1)
                        p_hvac_val = prob.var_dict[f"p_hvac_{zn}"].value.reshape(-1, 1)
                        p_hvac_val = np.concatenate((p_hvac_val, p_hvac_val[-1, :].reshape((1, 1))), axis=0)
                        variables.extend([t_in_val.flatten(), p_hvac_val.flatten()])
                        z.expected_results = pd.DataFrame(data=np.concatenate((t_in_val, p_hvac_val), axis=1),
                                                          index=self.ts_ems, columns=['Tin', 'P_hvac'])
                col_names = pd.MultiIndex.from_product([b.zones_df_no_plenum["name"], ['Tin', 'P_hvac']],
                                                       names=['zone', 'variable'])
                b.expected_results = pd.DataFrame(np.array(variables).T, index=self.ts_ems, columns=col_names)

                ### Save the results of the MILP
                self.expected_obj_val = obj_val
                self.cvxpy_solution = dict(zip(prob.var_dict.keys(), prob.solution.primal_vars.values()))
                self.expected_results = pd.DataFrame(data=np.concatenate((p_demand_max_vec, p_import_val, p_export_val),
                                                                         axis=1),
                                                     index=self.ts_ems[:-1],
                                                     columns=['P_demand_max', 'P_import', 'P_export'])
                self.solver_status = {'obj_val': obj_val,
                                      'solver_engine': solver_engine,
                                      'termination_condition': prob.status,
                                      'options': prob.solver_stats.extra_stats.Params,
                                      'mip_gap': mip_gap,
                                      'solving_time': solving_time,
                                      'status': prob.solver_stats.extra_stats,
                                      }

        # If a solution was not found, find if it is infeasible or unbounded
        else:
            self.feasible = False
            solver_parameters["verbose"] = True
            prob.solve(reoptimize=True, solver=solver_engine, **solver_parameters)
            self.cvxpy_solution, self.expected_results = None, None
            self.solver_status = {'obj_val': None,
                                  'solver_engine': solver_engine,
                                  'termination_condition': prob.status,
                                  'options': prob.solver_stats.extra_stats.Params,
                                  'mip_gap': None,
                                  'solving_time': 0,
                                  'status': prob.solver_stats.extra_stats,
                                  }

    def solve_cvxpy_mosek(self):
        # print(cp.installed_solvers())  # to check the installed solvers
        # # Print the constraints
        # print(prob.objective)
        # # Print the constraints
        # for constraint in prob.constraints:
        #     print(constraint)

        prob = self.cvxpy_opti_formulation
        solver_engine = cp.MOSEK
        solver_parameters = {'verbose': True, "mosek_params": {'MSK_DPAR_OPTIMIZER_MAX_TIME': 60,
                                                               'MSK_DPAR_MIO_TOL_REL_GAP': 0.01,
                                                               'MSK_IPAR_NUM_THREADS': 10}}
        # Attempt to solve the problem as long as it requires less than 6000s
        while solver_parameters["mosek_params"]["MSK_DPAR_OPTIMIZER_MAX_TIME"] <= 6000:
            try:
                prob.solve(solver=solver_engine, **solver_parameters)
                break
            except cp.error.SolverError:
                print(f"Problem not solved after {solver_parameters['mosek_params']['MSK_DPAR_OPTIMIZER_MAX_TIME']}s.")
                new_time_limit = solver_parameters["mosek_params"]["MSK_DPAR_OPTIMIZER_MAX_TIME"] * 10
                solver_parameters["mosek_params"]["MSK_DPAR_MIO_TOL_REL_GAP"] = 0.1
                if new_time_limit <= 6000:
                    print(f"Time limit increased to {new_time_limit}s.")
                else:
                    print("Impossible to find a solution for this problem")
                solver_parameters["mosek_params"]["MSK_DPAR_OPTIMIZER_MAX_TIME"] = new_time_limit



        print("status:", prob.status)
        ### analyze the solution and save the results
        # if the solver returned an inaccurate, solve again with a new solver
        # if prob.status == "optimal_inaccurate":
        #     # solver_engine = cp.MOSEK
        #     # solver_parameters = {'verbose': False, "mosek_params": {'MSK_DPAR_OPTIMIZER_MAX_TIME': 60,
        #     #                                                         'MSK_DPAR_MIO_TOL_REL_GAP': 0.01,
        #     #                                                         'MSK_IPAR_NUM_THREADS': 10}}
        #     # prob.solve(solver=solver_engine, **solver_parameters)
        #     solver_engine = cp.GUROBI
        #     solver_parameters = {'verbose': False, 'TimeLimit': 60, 'MIPGap': 0.01, 'Threads': 0}
        #     prob.solve(reoptimize=True, solver=solver_engine, **solver_parameters)
        #     print("status:", prob.status)

        # if there is a solution
        if prob.status == "optimal" or prob.status == "optimal_inaccurate":
            self.feasible = True
            obj_val = prob.value
            print(prob.solver_stats.extra_stats)
            mip_gap = prob.solver_stats.extra_stats["mip_gap"]
            solving_time = prob.solver_stats.solve_time
            # if the solution is optimal
            if prob.status == "optimal":
                print('* OPTIMISATION SUCCESSFUL *')
                print(f'Solving time = {solving_time:.2f} s')
                print("Solution value:\t", obj_val)
            # if the solution is suboptimal (or not guaranteed), stop due to time budget reached
            elif prob.status == "optimal_inaccurate":  # Gurobi returns user_limit
                print('* OPTIMISATION MET TERMINATION CRITERIA *')
                print(f'MIP gap = {mip_gap:.2%}')
                print("Solution value:\t", obj_val)

            p_import_val = prob.var_dict["p_import"].value.reshape(-1, 1)
            p_export_val = prob.var_dict["p_export"].value.reshape(-1, 1)
            if ((p_import_val * p_export_val).round(3) != 0).any():
                while True:
                    x = 0
                print(f"p_import_val: {p_import_val}")
                print(f"p_export_val: {p_export_val}")

                # raise ValueError('Warning: Import and export powers cannot be non-zero at the same time')
            p_demand_max_val = prob.var_dict["p_demand_max"].value
            p_demand_max_vec = np.full((self.nb_ts_ems - 1, 1), p_demand_max_val)

            # store the results
            for b in self.bldg_assets:
                bn = b.name
                variables = []
                for z in b.zone_assets:
                    if z.controlled:
                        zn = z.name
                        t_in_val = prob.var_dict[f"t_in_{zn}"].value.reshape(-1, 1)
                        p_hvac_val = prob.var_dict[f"p_hvac_{zn}"].value.reshape(-1, 1)
                        p_hvac_val = np.concatenate((p_hvac_val, p_hvac_val[-1, :].reshape((1, 1))), axis=0)
                        variables.extend([t_in_val.flatten(), p_hvac_val.flatten()])
                        z.expected_results = pd.DataFrame(data=np.concatenate((t_in_val, p_hvac_val), axis=1),
                                                          index=self.ts_ems, columns=['Tin', 'P_hvac'])
                col_names = pd.MultiIndex.from_product([b.zones_df_no_plenum["name"], ['Tin', 'P_hvac']],
                                                       names=['zone', 'variable'])
                b.expected_results = pd.DataFrame(np.array(variables).T, index=self.ts_ems, columns=col_names)

                ### Save the results of the MILP
                self.expected_obj_val = obj_val
                self.cvxpy_solution = dict(zip(prob.var_dict.keys(), prob.solution.primal_vars.values()))
                self.expected_results = pd.DataFrame(data=np.concatenate((p_demand_max_vec, p_import_val, p_export_val),
                                                                         axis=1),
                                                     index=self.ts_ems[:-1],
                                                     columns=['P_demand_max', 'P_import', 'P_export'])
                self.solver_status = {'obj_val': obj_val,
                                      'solver_engine': solver_engine,
                                      'termination_condition': prob.status,
                                      'options': solver_parameters,
                                      'mip_gap': mip_gap,
                                      'solving_time': solving_time,
                                      'status': prob.solver_stats.extra_stats,
                                      }

        # If a solution was not found, find if it is infeasible or unbounded
        else:
            self.feasible = False
            solver_parameters["verbose"] = True
            # prob.solve(reoptimize=True, solver=solver_engine, **solver_parameters)
            self.cvxpy_solution, self.expected_results = None, None
            self.solver_status = {'obj_val': None,
                                  'solver_engine': solver_engine,
                                  'termination_condition': prob.status,
                                  'options': solver_parameters,
                                  'mip_gap': None,
                                  'solving_time': solver_parameters["mosek_params"]["MSK_DPAR_OPTIMIZER_MAX_TIME"],
                                  'status': {},
                                  }

    def get_cvxpylayer_parameters(self, cvxpylayer_opti_formulation, relu_relaxation_for_convex_opti):
        """
        Get all the parameters of the cvxpylayer. These are the parameters of the optimisation problem + the binaries
        :param cvxpylayer_opti_formulation:
        :param relu_relaxation_for_convex_opti: a string indicating how the ReLU binaries are formulated.
                  It can be "fixed_bin" or "qp". If "fixed_bin", the binaries are fixed at the optimum of the MIQP
                  so they are parameters. If "qp", the binaries are continuous variables between 0 and 1.
        :return: parameter_init: dict of all the parameters of the cvxpylayer (including the binaries)
                    parameters_to_optimize: dict of the parameters to optimize by the DFL (only the weights and biases)
        """
        ### initialize the parameters
        prob = self.cvxpy_opti_formulation
        parameter_init = {}  # all the parameters of the cvxpylayer
        for k, p in prob.param_dict.items():
            requires_grad = False
            if any(sb in k for sb in ["weights", "biases", "rc_alpha", "rc_inv_R", "rc_inv_C", "rc_h_eff", "rc_c_eff"]):
                requires_grad = True
            parameter_init[k] = torch.tensor(p.value, requires_grad=requires_grad, dtype=torch.float)
        # only the parameters to be optimized by the DFL (gradient = True)
        parameters_to_optimize = {k: p for k, p in parameter_init.items() if p.requires_grad}
        # recover the binary values at the optimum of the MILP if cvxpylayer formulation is fixed_bin
        if relu_relaxation_for_convex_opti == "fixed_bin":
            binaries = {k.replace("binary", "binary_dpp"): torch.tensor(v.value, requires_grad=False, dtype=torch.float)
                        for k, v in prob.var_dict.items() if "binary" in k}
            parameter_init.update(binaries)
        # gather the list of names of the variables and parameters to check they match
        parameter_names = list(parameter_init.keys())
        cvxpylayer_parameter_names = list(cvxpylayer_opti_formulation.param_dict.keys())
        if cvxpylayer_parameter_names != parameter_names:
            debug = list(map(lambda x, y: x != y, cvxpylayer_parameter_names, parameter_names))
            print(debug)
            raise ValueError("The parameters of the cvxpylayer do not match the parameters of the built list of"
                             " parameters.")

        return parameter_init, parameters_to_optimize


    def expost_simulation(self):
        ep = EnergyPlusSimulator()
        for b in self.bldg_assets:
            b.idf_filepath_expost = b.modify_idf(self.T0)
            simulationvariables = get_variable_list(b)
            # add the temperature setpoints to the list of variables
            for z in b.zone_assets[[("Plenum" not in z.name and "Attic" not in z.name) for z in b.zone_assets]]:
                simulationvariables.append(
                    f"Actuator,Zone Temperature Control,Cooling Setpoint,{z.name.upper()};")
                simulationvariables.append(
                    f"Actuator,Zone Temperature Control,Heating Setpoint,{z.name.upper()};")
            # remove duplicates
            simulationvariables = list(dict.fromkeys(simulationvariables))
            # In this case, desired frequence is the desired frequency for the inputs of the simulation
            # it must be the one of the ems
            b.set_simulationvariables(simulationvariables, simulationfrequency=self.dt_sim,
                                      desiredfrequency=self.dt_ems)
            b.fullsimulation_expost, b.nb_warmupts_expost = ep.run_simulation(b, 3,
                                                                              ep.callback_temperature_control,
                                                                              b.idf_filepath_expost)
            os.unlink(b.idf_filepath_expost)
            b.simulation_expost = b.fullsimulation_expost.loc[~b.fullsimulation_expost["warmup"]]
            # set the dates as index (avoid SettingWithCopyWarning)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                b.simulation_expost["Timestep"] = b.simulation_expost["Timestep"].apply(
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
            # b.simulation_expost = b.zonal_PACU_electricity_energy(b.simulation_expost)
            # set the expost Tin and P_hvac in each zone
            b.set_expost()

    def compute_power_cost(self, expost_or_expected):
        """
        Compute the objective value of the ex-post problem.
        :param expost_or_expected: a string indicating if the expected or the ex-post results are considered
        :param with_nd_load: boolean indicating if the non-dispatchable load is considered or not
        :return: the corrected prices considering the demand charge
        """
        nd_load_ttl = pd.Series(0, index=self.ts_ems[:-1])
        p_hvac_ttl = pd.Series(0, index=self.ts_ems[:-1])
        for b in self.bldg_assets:
            nd_load_ttl += b.nd_load.loc[self.ts_ems[:-1]]
            if expost_or_expected == "expost":
                p_hvac_ttl += b.expost_results.loc[:, (slice(None), "P_hvac")].resample(f"{self.dt_ems}H").sum().sum(
                    axis=1) / 1000
            elif expost_or_expected == "expected":
                p_hvac_ttl += b.expected_results.loc[:, (slice(None), "P_hvac")].sum(axis=1)
        p_import = p_hvac_ttl  # N.B.: in this case, p_import(here) = p_import - p_export in opti
        if self.nd_load_flag:
            p_import += nd_load_ttl
        max_import_ts = p_import == p_import.max()
        prices = self.market.prices_import.loc[self.ts_ems[:-1]]
        increased_prices = prices + self.market.demand_charge
        corrected_prices = np.where(max_import_ts, increased_prices, prices)
        obj_value = np.sum(p_import * corrected_prices)
        return obj_value, corrected_prices


    def compute_temperature_penalty(self, expost_or_expected):
        """
        Compute the penalty for the temperature constraint.
        :param self:
        :param expost_or_expected: a string indicating if the expected or the ex-post results are considered
        :return: the penalty for the temperature constraint
        """
        penalty = 0
        for b in self.bldg_assets:
            for z in b.zone_assets:
                if z.controlled:
                    if expost_or_expected == "expected":
                        Tin = z.expected_results.loc[:, "Tin"]
                        Ttgt = z.Ttgt.loc[self.ts_ems]
                        ow = z.occupancy_weights
                    elif expost_or_expected == "expost":
                        Tin = z.expost_results.loc[:, "Tin"]
                        Ttgt = z.Ttgt.loc[self.ts_ems].resample(f"{self.dt_ems}H").interpolate("time")
                        ow = z.occupancy_weights.resample(f"{self.dt_ems}H").interpolate("time")
                    penalty += (ow * (Tin - Ttgt) ** 2).sum()
        return penalty


    def solve_ems_and_simulate_expost(self, solver_parameters):
        """
        return the time to solve the ems problem and simulate the ex-post results
        """
        opti_time, simu_time = Stopwatch(), Stopwatch()
        # solve the ems
        opti_time.start()
        self.solve_cvxpy_gurobi(solver_parameters)
        opti_time.stop()
        # if the problem is feasible: compute and save the expected and ex-post costs
        if self.feasible:
            # compute the expect cost and expected temperature penalty
            self.expected_power_cost, _ = self.compute_power_cost("expected")
            self.expected_temperature_penalty = self.compute_temperature_penalty("expected")
            # simulate the decisions
            simu_time.start()
            self.expost_simulation()
            simu_time.stop()
            # compute the ex-post cost and ex-post temperature penalty
            self.expost_power_cost, _ = self.compute_power_cost("expost")
            self.expost_temperature_penalty = self.compute_temperature_penalty("expost")

        return opti_time.get_elapsed_time(), simu_time.get_elapsed_time()

    def plot_results(self, ncols, nrows=None, path=None, format='svg', savefig=False, showfig=True):
        # PLOT 1: plot the price
        fig, ax = plt.subplots(1, 1)
        myFmt = mdates.DateFormatter('%H:%M')  # here you can format your datetick labels as desired
        plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=5))
        plt.gca().xaxis.set_major_formatter(myFmt)
        ax.plot(self.ts_ems, self.market.prices_import.loc[self.ts_ems],
                label='ToU price')
        ax.plot(self.ts_ems, self.market.prices_export.loc[self.ts_ems],
                label='Feed-in price')
        ax.set_ylim(top=max(self.market.prices_import.max(), self.market.prices_export.max()) * 1.25)
        ax.set_xlim(left=self.ts_ems[0], right=self.ts_ems[-1])
        ax.set_ylabel('Price [€/kWh]')
        ax.set_xlabel('Time')
        ax.legend(loc="upper left")
        fig.subplots_adjust(bottom=0.15)
        if savefig:
            # pass
            fig.savefig(os.path.join(path, f"Prices.{format}"))
        fig.show() if showfig else plt.close(fig)

        for b in self.bldg_assets:
            # controllable zones
            controllable_zones = [cz for cz in b.zone_assets if cz.controlled]
            # PLOT 2: Create the temperature plot
            nrows = int(np.ceil(len(controllable_zones) / ncols)) if nrows is None else nrows
            fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey=True, figsize=(2 * ncols, 2.4 * nrows))
            axes = axes.flatten()
            # Avoid used warning because of the "_" at the beginning of the legend label
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for i, z in enumerate(controllable_zones):
                    myFmt = mdates.DateFormatter('%H:%M')  # here you can format your datetick labels as desired
                    plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=4))
                    plt.gca().xaxis.set_major_formatter(myFmt)
                    # to display the legend only once, set an underscore to the label if i != 0
                    us = "_" if i != 0 else ""
                    z.expected_results.plot(y=['Tin'], label=[f"{us}" + "Expected Ind. Temp."], ax=axes[i],
                                            x_compat=True,
                                            legend=False)
                    z.Tmin.loc[self.ts_ems].plot(label=f"{us}" + "Heating Setpt", x_compat=True,
                                                 ax=axes[i])
                    z.Tmax.loc[self.ts_ems].plot(label=f"{us}" + "Cooling Setpt", x_compat=True,
                                                 ax=axes[i])
                    # plot expost temperature
                    b.simulation_expost.loc[self.ts_sim,
                    f"Zone Mean Air Temperature,{z.name}"].plot(
                        label=f"{us}" + "Ex-post Ind. Temp.", ax=axes[i], x_compat=True, legend=False)
                    # other plot settings
                    axes[i].set_title(z.name)
                    axes[i].set_ylabel('Temperature [°C]')
                    axes[i].set_xlabel('Time')
            # fig.subplots_adjust(bottom=0.25, top=0.9)
            fig.legend(loc='lower center', bbox_to_anchor=(0.5, -0.1), ncols=4)
            fig.suptitle(f"Zonal temperature - {self.T0.strftime('%Y-%m-%d')}")
            fig.autofmt_xdate()
            fig.tight_layout()
            if savefig:
                fig.savefig(os.path.join(path, f"Temperatures_{b.name}.{format}"))
            fig.show() if showfig else plt.close(fig)

            # PLOT 3: Create the regularized temperature plot
            fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey=True, figsize=(2 * ncols, 2.4 * nrows))
            axes = axes.flatten()
            custom_cmap = plt.get_cmap('Blues')
            custom_colors = custom_cmap(np.linspace(0, 1, 8))
            # Avoid used warning because of the "_" at the beginning of the legend label
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for i, z in enumerate(controllable_zones):
                    myFmt = mdates.DateFormatter('%H:%M')  # here you can format your datetick labels as desired
                    plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=4))
                    plt.gca().xaxis.set_major_formatter(myFmt)
                    # to display the legend only once, set an underscore to the label if i != 0
                    us = "_" if i != 0 else ""
                    z.expected_results.plot(y=['Tin'], ax=axes[i], c='r', ls="--", label=[f"{us}Ind. Temp."],
                                            x_compat=True, legend=False)
                    # compute the shaded area
                    x = z.expected_results.index
                    t_rule = self.cvxpy_opti_formulation.param_dict.get(f"temp_rule_{z.name}").value
                    w = z.occupancy_weights.loc[x].values
                    axes[i].hlines(y=t_rule, xmin=x[0], xmax=x[-1], colors='k', linestyles="solid", label=f"{us}Rule")
                    for j, c in enumerate(np.flip(np.concatenate(([1], range(5, 26, 5)), axis=0))):
                        delta_t = np.sqrt(c / w)
                        y1 = t_rule - delta_t
                        y2 = t_rule + delta_t
                        axes[i].fill_between(x, y1, y2, color=custom_colors[1+j], label=f"{us}c={c}")
                    # other plot settings
                    axes[i].margins(x=0)
                    axes[i].set_title(z.name)
                    axes[i].set_ylabel('Temperature [°C]')
                    axes[i].set_xlabel('Time')
                    # makes sure that the axis ticks are on top of the color patches
                    axes[i].set_axisbelow('line')
            fig.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), ncols=4)
            fig.suptitle(f"Zonal temperature - {self.T0.strftime('%Y-%m-%d')}")
            fig.autofmt_xdate()
            fig.tight_layout()
            if savefig:
                fig.savefig(os.path.join(path, f"TemperaturesRegularized_{b.name}.{format}"))
            fig.show() if showfig else plt.close(fig)

            # PLOT 4: Create the HVAC plot
            nrows = int(np.ceil(len(controllable_zones) / ncols)) if nrows is None else nrows
            fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey=True, figsize=(2 * ncols, 2.4 * nrows))
            axes = axes.flatten()
            # Avoid used warning because of the "_" at the beginning of the legend label
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for i, z in enumerate(controllable_zones):
                    myFmt = mdates.DateFormatter('%H:%M')  # here you can format your datetick labels as desired
                    plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=4))
                    plt.gca().xaxis.set_major_formatter(myFmt)
                    # to display the legend only once, set an underscore to the label if i != 0
                    us = "_" if i != 0 else ""
                    z.expected_results.plot(y=['P_hvac'], label=[f"{us}Expected Power"], ax=axes[i], x_compat=True,
                                            legend=False)
                    # plot expost power (conversion necessary)
                    axes[i].plot(self.ts_sim,
                                 b.simulation_expost.loc[
                                     self.ts_sim, f"Air System Electricity Energy,{z.acu}[Wh]"] / self.dt_sim / 1000,
                                 label=f"{us}Ex-post Power")
                    axes[i].hlines(z.hvac_capacity, self.ts_ems[0], self.ts_ems[-1], colors="C2", label=f"{us}Capacity")
                    axes[i].set_title(z.name)
                    axes[i].set_ylabel('HVAC Power [kWe]')
                    axes[i].set_xlabel('Time')
            fig.legend(loc='lower center', bbox_to_anchor=(0.5, -0.1), ncols=3)
            fig.suptitle(f"Zonal HVAC power - {self.T0.strftime('%Y-%m-%d')}")
            fig.autofmt_xdate()
            fig.tight_layout()
            if savefig:
                fig.savefig(os.path.join(path, f"PowerHVAC_{b.name}.{format}"))
            fig.show() if showfig else plt.close(fig)

            # PLOT 5: Aggregated HVAC Power
            fig, ax = plt.subplots(1, 1)
            myFmt = mdates.DateFormatter('%H:%M')  # here you can format your datetick labels as desired
            plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=4))
            plt.gca().xaxis.set_major_formatter(myFmt)
            b.expected_results.loc[self.ts_ems, (slice(None), "P_hvac")].sum(axis=1).plot(label='Expected Power', ax=ax,
                                                                                          legend=False, x_compat=True)
            ax.plot(self.ts_sim,
                    b.simulation_expost.filter(like=f"Air System Electricity Energy").loc[self.ts_sim].sum(axis=1).div(
                        self.dt_sim * 1000),
                         label=f"Ex-post Power")
            ax.set_ylabel('HVAC Power [kWe]')
            ax.set_xlabel('Time')
            fig.legend(loc='lower center', bbox_to_anchor=(0.5, -0.1), ncols=3)
            fig.suptitle(f"Building HVAC power - {self.T0.strftime('%Y-%m-%d')}")
            fig.autofmt_xdate()
            fig.tight_layout()
            if savefig:
                fig.savefig(os.path.join(path, f"AggregatedPowerHVAC_{b.name}.{format}"))
            fig.show() if showfig else plt.close(fig)

            plt.close('all')


    # def save_results(self, summary_filepath):
    #     """
    #
    #     :param summary_filepath: an xlsx file which contains the summary of the results
    #     :return:
    #     """
    #     # Append those results to the summary file
    #     if os.path.exists(summary_filepath):
    #         summary_df = pd.read_excel(summary_filepath, index_col=0, header=0)
    #     else:
    #         summary_df = pd.DataFrame(
    #             columns=['Date', 'Cluster', 'Net. Constr.', 'N_features', 'Resp. Var.', 'Model Type',
    #                      'Model carac.', 'Building List', ])
    #
    #     if self.solver_status['solver_engine'] == "GUROBI":
    #         summary_df = pd.concat(
    #             [summary_df, pd.DataFrame({'Date': datetime.now(),
    #                                        'Cluster': day_type,
    #                                        'Model Type': full_t_in_model_type,
    #                                        'Model carac.': self.bldg_assets[0].ml_model_doc,
    #                                        'N_ts': self.nb_ts_ems,
    #                                        'solver': self.solver_status['solver_engine'],
    #                                        'Solving time': self.solver_status['solving_time'],
    #                                        'MIP gap': self.solver_status['mip_gap'],
    #                                        'Termination condition': self.solver_status['termination_condition'],
    #                                        'N_cont_var': self.solver_status[
    #                                            'status'].problem.number_of_continuous_variables,
    #                                        'N_int_var': self.solver_status['status'].problem.number_of_integer_variables,
    #                                        'N_bin_var': self.solver_status['status'].problem.number_of_binary_variables,
    #                                        'N_constraints': self.solver_status['status'].problem.number_of_constraints,
    #                                        'Electricity Cost': self.solver_status['obj_val'],
    #                                        }, index=[summary_df.shape[0]])
    #              ], axis=0)
    #     elif self.solver_status['solver_engine'] == "MOSEK":
    #         summary_df = pd.concat(
    #             [summary_df, pd.DataFrame({'Date': datetime.now(),
    #                                        'Cluster': day_type,
    #                                        'Model Type': full_t_in_model_type,
    #                                        'Model carac.': self.bldg_assets[0].ml_model_doc,
    #                                        'N_ts': self.nb_ts_ems,
    #                                        'Solving time': self.solver_status['solving_time'],
    #                                        'MIP gap': self.solver_status['status']['mip_gap'],
    #                                        'Termination condition': self.solver_status['termination_condition'],
    #                                        'N_cont_var': self.solver_status[
    #                                            'status']['nb_var_cont'],
    #                                        'N_int_var': self.solver_status['status']['nb_var_int'],
    #                                        'N_bin_var': self.solver_status['status']['nb_var_bin'],
    #                                        'N_constraints': self.solver_status['status']['nb_const'],
    #                                        'N_parameters': self.solver_status['status']['nb_param'],
    #                                        'Electricity Cost': self.solver_status['obj_val'],
    #                                        }, index=[summary_df.shape[0]])
    #              ], axis=0)
    #     summary_df.to_excel(summary_filepath, index=True)

