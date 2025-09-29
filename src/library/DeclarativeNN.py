"""
Author: Pietro Favaro
Date: April 28, 2024
Description: This file contains the implementation of a Declarative Neural Network (DNN) class.
        At the moment, the DNN is made of one cvxpy layer.
"""

from cvxpylayers.torch import CvxpyLayer
from itertools import product
import matplotlib.pyplot as plt
from src.library.DeclarativeNNcommon import *
from src.library.Classes import Stopwatch
from src.library.EnergyManagementSystem import EMS
import warnings

# Set the default plt.show to 100 dpi but savefig to 600 to fit IEEE standards
plt.style.use(['science', 'ieee', 'grid', 'no-latex'])
plt.rcParams.update({'figure.dpi': '100', 'savefig.dpi': '600', "savefig.format": 'pdf',
                     'axes.xmargin': 0.05, 'axes.ymargin': 0.05})


# plt.rcParams.update({'axes.prop_cycle': cycler(color="tab20")})


class DeclarativeNN():
    """
        Class which implements a Declarative Neural Network (DNN) made of one cvxpy layer.

        Attributes:
            paths: dictionary, the paths of the different folders
            ems_qp: EMS object, the energy management system relaxed as a qp
            ems_miqp: EMS object, the energy management system as it should be (without the need to differentiate it)
            smiqp_param_sd: tuple (weights, biases), size 2, containing a tuple (sample, distribution), size 2
                of the nn parameters fed to the miqp problem in stochastic smoothing
            ems_cvxpylayer: EMS object, the energy management system adapted to be differentiable in the cvxpylayer
            with the Cvxpylayer object as attribute (EMS becomes a wrapper for Cvxpylayer object)
            cvxpylayer_opti_formulation: cp.Problem, the cvxpy formulation of the optimization problem
            cvxpylayer: CvxpyLayer object, the cvxpy layer of the DNN
            cvxpylayer_parameters: list of dictionaries. All the names of the parameters of the cvxpy layer as
                            keys and the corresponding tensor as values.
            param_to_optimize: dictionary, the names of the parameters of the cvxpy layer to be optimized
                            (i.e., parameter tensors with gradient) as keys and the corresponding tensor as values
            param_to_optimize_std: dictionary, the standard deviation of the parameters to be optimized
            cvxpylayer_solution: torch.Tensor, the solution of the cvxpy layer
            cvxpylayer_variable_names: list of string, the names of the variables of the cvxpy layer (the output of the layer)
            optimizer: torch.optim.Optimizer, the optimizer of the DNN
            scheduler: torch.optim.lr_scheduler, the learning rate scheduler of the DNN

    """

    def __init__(self):
        # attributes
        self.paths = {}
        self.ems_qp = None
        self.ems_miqp = None
        self.smiqp_param_sd = None
        # self.cvxpylayer = None
        # self.cvxpylayer_opti_formulation = None  # not needed because cvxpylayer because an ems
        # self.cvxpylayer_parameters = None  # all the parameters of the cvxpylayer problem
        self.param_to_optimize = None
        self.param_to_optimize_std = None
        self.cvxpylayer_solution = None
        self.cvxpylayer_variable_names = None
        self.optimizer = None
        self.scheduler = None
    
    @property
    def cvxpylayer_param_to_optimize(self):
        """
        getter for the cvxpylayer_param_to_optimize attribute. It is the same information as in the param_to_optimize
        but formulated for the cvxpylayer.
        Returns:
            the cvxpylayer_param_to_optimize attribute
        """
        d = {}
        for i, p in enumerate(self.param_to_optimize):  # p is a dictionary (building name -> layer name -> weights/biases)
            w_or_b = "weights" if i == 0 else "biases"
            for bn, bd in p.items():  # bd is a dictionary (layer name -> weights/biases); the dictionary containing the parameters for a given building model
                for k, v in bd.items():  # v is the leaf parameter (weights/biases)
                    if self.ems_miqp.thermal_model == "nn":
                        d[f"nn_layer_{k}_{w_or_b}"] = v
                    elif self.ems_miqp.thermal_model == "spatialrcmodel":
                        for param, vv in v.items():
                            if "alpha" in param:
                                for zn, alpha in vv.items():
                                    d[f"rc_{param}_{k}_x_{zn}"] = alpha
                            elif "R" in param or "C" in param:
                                d[f"rc_inv_{param}_{k}"] = 1/vv
                            else:  # h_eff or c_eff
                                d[f"rc_{param}_{k}"] = vv
                    elif self.ems_miqp.thermal_model == "rcmodel":
                        for param, vv in v.items():
                            if "R" in param or "C" in param:
                                d[f"rc_inv_{param}_{k}"] = 1/vv
                            else:  # h_eff or c_eff
                                d[f"rc_{param}_{k}"] = vv

                    else:
                        raise ValueError(f"Thermal model {self.ems_miqp.thermal_model} not recognized."
                                         f"Thermal model must be either 'nn' or 'rcmodel'.")
        return d

    def get_param_to_optimize(self, requires_grad: bool):
        if requires_grad:
            return self.param_to_optimize
        else:
            return self.map_param_to_optimize(lambda x: x.detach().clone(), inplace=False)

    def get_param_to_optimize_for_optimizer(self):
        t_l = []
        self.map_param_to_optimize(c.append_and_return, t_l, inplace=False)
        self.map_param_to_optimize_std(c.append_if_leaf_rg_and_return, t_l, inplace=False)
        return t_l

    def map_param_to_optimize(self, f, *args, inplace=False, **kwargs):
        """
        Apply a function to the leaf values param_to_optimize attribute.
        Args:
            f:

        Returns:

        """
        return tuple([c.iter_dic(p, f, *args, inplace=inplace, **kwargs) for p in self.param_to_optimize])


    def map_param_to_optimize_std(self, f, *args, inplace=False, **kwargs):
        """
        Apply a function to the leaf values param_to_optimize attribute.
        Args:
            f:

        Returns:

        """
        return tuple([c.iter_dic(p, f, *args, inplace=inplace, **kwargs) for p in self.param_to_optimize_std])


    def set_param_to_optimize(self, parameters):
        """
        Setter for the param_to_optimize attribute.
        :param parameters: tuple (weights, biases) containing dictionaries (building name -> layer name -> weights/biases)
        """
        self.param_to_optimize = parameters
        # the parameters are saved with gradient tracking per default
        self.param_to_optimize = self.map_param_to_optimize(lambda x: x.requires_grad_())


    def set_param_to_optimize_std(self, std_w, std_b, requires_grad):
        """
        Setter for the param_to_optimize_std attribute. Same structure as param_to_optimize but with the standard
        deviation of the parameters to optimize.
        :param std_w: float or fct, indicating the standard deviation of the weights
                fct are applied w.r.t. the initial value of the weights. E.g., std_w = lambda x: 0.1 * x
                implies that the standard deviation of the weights is 10% of the initial value of the weights
        :param std_b: float or fct, indicating the standard deviation of the biases
        """
        param_to_optimize_std = [None] * len(self.param_to_optimize)  # 0 for weights, 1 for biases
        for i, std in enumerate((std_w, std_b)):  # i=0 for weights, i=1 for biases
            if callable(std):
                # adapt std to the requires_grad attribute
                def std_grad(x: torch.Tensor) -> torch.Tensor:
                    """
                    if requires_grad is False, overwrite std to be applied on the detached tensor of the parameter
                    """
                    if requires_grad:
                        # per default, std is applied on the parameter with gradient tracking so its gradient is tracked
                        return std(x)
                    else:
                        # if std_requires_grad is False, std is applied on the detached tensor of the parameter
                        return std(x.detach())

                param_to_optimize_std[i] = c.iter_dic(self.param_to_optimize[i], std_grad, inplace=False)
            elif isinstance(std, (int, float)):
                param_to_optimize_std[i] = c.iter_dic(self.param_to_optimize[i],
                                                           lambda x: torch.full_like(x, std, requires_grad=requires_grad),
                                                      inplace=False)
            else:
                raise TypeError("std_w and std_b must be either float, int, or callable.")
        self.param_to_optimize_std = tuple(param_to_optimize_std)


    def build_cvxpylayer(self, ems, thermal_model, ems_relaxation):
        # Build the opti formulation of the declarative layer
        prob = ems.build_cvxpy(thermal_model=thermal_model, relu_relaxation=ems_relaxation)
        self.cvxpylayer_opti_formulation = prob

        # gather cvxpylayer variables (the output)
        variables, variable_names = [], []
        for k, v in prob.var_dict.items():
            if k.startswith(("t_in", "p_hvac")):
                variables.append(v)
                variable_names.append(k)

        # create the declarative layer
        prob_param_dict = prob.param_dict  # get all the parameters of the problem as input (no choice, forced)

        self.cvxpylayer = CvxpyLayer(prob, parameters=list(prob_param_dict.values()), variables=variables)
        self.cvxpylayer_variable_names = variable_names

    def get_w_and_b(self, thermal_model, ems, fct=c.detach_clone):
        """
        function to return a copy of the updated weights and biases of the cvxpylayer network
        :param fct: function to apply to each matrix of weights and biases. Per default, it detaches and copy the tensor
        :return: the new elements to be appended to the nn_weights, and nn_biases lists
        """

        if thermal_model == "nn":
            # get the updated weights and biases
            w_dic = {b.name: {l: fct(self.param_to_optimize.get(f"nn_layer_{l}_weights"))
                              for l, _ in enumerate(b.nn.layers) if l > 0}
                     for b in ems.bldg_assets}
            b_dic = {b.name: {l: fct(self.param_to_optimize.get(f"nn_layer_{l}_biases"))
                              for l, _ in enumerate(b.nn.layers) if l > 0}
                     for b in ems.bldg_assets}

        # for the rc model
        elif thermal_model == "rcmodel":
            w_dic, b_dic = {}, {}
            # for each building
            for b in ems.bldg_assets:
                bn = b.name
                w_dic[bn], b_dic[bn] = {}, {}
                # for each zone
                for zn in b.zones_df_no_plenum["name"]:
                    alpha = {zzn: fct(self.param_to_optimize.get(
                        f"rc_alpha_{zn}_x_{zzn}")) for zzn in b.zones_df_no_plenum["name"]}
                    h_eff = fct(self.param_to_optimize.get(f"rc_h_eff_{zn}"))
                    c_eff = fct(self.param_to_optimize.get(f"rc_c_eff_{zn}"))
                    R = 1 / fct(self.param_to_optimize.get(f"rc_inv_R_{zn}"))
                    C = 1 / fct(self.param_to_optimize.get(f"rc_inv_C_{zn}"))
                    # store a dictionary with the parameters of the rc model
                    w_dic[bn][zn] = {"alpha": alpha, "h_eff": h_eff, "c_eff": c_eff, "R": R, "C": C}
                    b_dic[bn][zn] = {}
        else:
            raise ValueError("The thermal model must be either 'nn' or 'rcmodel'.")

        return w_dic, b_dic

    def step(self, l):
        """
        Make one gradient descent step for the given loss function.
        :param l: the loss associated with one batch
        :return:
        """
        # Resets the gradients of all optimized torch.Tensors to 0
        self.optimizer.zero_grad()
        # compute the gradient of Tensor l (the value of the loss) across the graph
        l.backward()
        # update the parameters by performing one step in the sens of the gradient computed before
        self.optimizer.step()


    def step_ss(self, l):
        """
        Make one gradient descent step with REINFORCE (stochastic smoothing) for the given reward l.
        :param l: the task loss (or the reward in RL)  associated with one batch
        of the parameters samples that lead to this reward
        :return:
        """
        # the final loss is computed using reinforce:
        # https://pytorch.org/docs/stable/distributions.html#score-function
        # there is no more gradient link between l and the parameters (RL training)
        # the gradient step is independent for each parameter

        def backprop(s, d):
            """
            Function to backpropagate the loss for a given sample and distribution.
            """
            # Build the loss for the whole layer
            loss = d.log_prob(s) * l
            # Resets the gradients of all optimized torch.Tensors to 0
            self.optimizer.zero_grad()
            # compute the gradient
            loss.mean().backward(retain_graph=True)
            # update the parameters by performing one step in the sens of the gradient computed before
            # the learning rate is handled by the optimizer
            self.optimizer.step()


        def recursive_backprop(s, d):
            """
            Function to backpropagate the loss for a given sample and distribution.
            It handles the case where s and d are dictionaries.
            """
            if isinstance(s, dict) and isinstance(d, dict):
                for (_, ss), (_, dd) in zip(s.items(), d.items()):
                    recursive_backprop(ss, dd)
            else:
                backprop(s, d)

        for param_sd in self.smiqp_param_sd:  # extract the (sample, distrib) tuple for the weights, then for the biaises
            param_s = param_sd[0]
            param_d = param_sd[1]
            recursive_backprop(param_s, param_d)

    def run_ems(self, epoch, medoid_name, ems_dic, dataset_str, day_date, thermal_model, ems_relaxation):
        """
        Run the EMS for a given day and medoid based on all the information of the training
        Args:
            init_flag:
            epoch:
            medoid_name:
            ems_dic:
            dataset_str: "training", "validation" or "test"
            day_date:
            thermal_model:
            paths:
            now:
            ems_relaxation:
            warm_start:
            hyperparameters:
            snr:

        Returns: the ems set-up, solved and recorded

        """
        # check values
        if dataset_str in ["validation", "test"]:
            if ems_relaxation != "miqp":
                warnings.warn(
                    f"Validation and test should be performed with the 'ems_relaxation = miqp' (i.e. no relaxation).")

        ### SOLVE THE TRUE MIQP
        if ems_relaxation == "fixed_bin" or ems_relaxation == "miqp" or ems_relaxation == "ss":
            # if not init_flag:
            #     # rebuild the true ems (ems_miqp) with the updated parameters
            #     self.ems_miqp, _, _, _ = EMS.build_ems(day_date, thermal_model, self.paths, now,
            #                                       self.ems_miqp.cvxpy_opti_formulation, "miqp",
            #                                       warm_start, hyperparameters, snr)
            self.ems_miqp.update_ems(day_date)

            # Apply noise to the parameters
            if ems_relaxation == "ss":
                self.smiqp_param_sd = wrapper_normal_distribution(self.param_to_optimize, self.param_to_optimize_std)
                cvxpy_param = (self.smiqp_param_sd[0][0], self.smiqp_param_sd[1][0])
            else:
                cvxpy_param = self.param_to_optimize

            # Set the ems parameters
            self.ems_miqp.set_cvxpy_parameters(*cvxpy_param, nd_load_flag=False)
            if thermal_model == "nn":
                self.ems_miqp.set_zhat_and_z_bounds()
            # solve and record the ems
            if dataset_str == "test":
                # Time limit for the test is 1h
                opti_time, simu_time = self.ems_miqp.solve_ems_and_simulate_expost(
                    {'verbose': False, 'TimeLimit': 1200, 'MIPGap': 0.01, 'Threads': 0})
                ems_dic[dataset_str].at[0, medoid_name] = deepcopy(self.ems_miqp)
            else:
                opti_time, simu_time = self.ems_miqp.solve_ems_and_simulate_expost(
                    {'verbose': False, 'TimeLimit': 60, 'MIPGap': 0.01, 'Threads': 0})
                ems_dic[dataset_str].at[epoch, medoid_name] = deepcopy(self.ems_miqp)
            return self.ems_miqp, (opti_time, simu_time)
        ### SOLVE THE QP RELAXATION FOR CONVEXITY
        elif ems_relaxation == "qp":
            # cvxpy_opti_formulation = None if init_flag else self.ems_qp.cvxpy_opti_formulation
            # self.ems_qp, _, _, _ = EMS.build_ems(day_date, thermal_model, self.paths, now,
            #                                 cvxpy_opti_formulation, "qp",
            #                                 warm_start, hyperparameters, snr)
            self.ems_qp.update_ems(day_date)
            # Set the ems optimization parameters
            self.ems_qp.set_cvxpy_parameters(*self.param_to_optimize, nd_load_flag=False)
            # solve the ems (even though it is not necessary in this case but it simplifies the code)
            opti_time, simu_time = self.ems_qp.solve_ems_and_simulate_expost(
                {'verbose': False, 'TimeLimit': 60, 'Threads': 0})

            # record the ems
            ems_dic[dataset_str].at[epoch, medoid_name] = deepcopy(self.ems_qp)
            return self.ems_qp, (opti_time, simu_time)
        else:
            raise ValueError(f"Unknown relaxation for the ems optimization: {ems_relaxation}."
                             f" Only 'fixed_bin', 'miqp', and 'qp' are supported.")


    def create_datasets(self, trng_seed: int, val_test_seed: int):
        """
        Args:
            trng_seed:
                Seed for the random number generator (for shuffling the training set)
                If None or equal to 0, no shuffling is applied
            val_test_seed:
                Seed for the random number generator (for selecting the validation and test days)
                Should always be the same to have comparable results across trainings

        Returns:
            val_dates:
                Series with the validation dates for each medoid
            test_dates:
                Series with the test dates for each medoid
        """
        ### training set
        # Read the medoids days and reorder them to create a smooth transition between the days
        folderpath = "data/optimization_parameters/2006-2020"
        medoid_labels = pd.read_csv(os.path.join(folderpath, "Labels10Days.csv"), index_col=0,
                                    parse_dates=True)
        medoids = pd.read_csv(os.path.join(folderpath, "Clusters10Days.csv"), index_col=0, parse_dates=True)
        medoids = medoids.sort_values(by="Mean Tamb", ascending=True)
        m_idx = medoids.index
        nb_medoids = len(m_idx)
        cycle = [m_idx[i] for i in range(nb_medoids) if i % 2 == 0] + [m_idx[i] for i in range(nb_medoids, -1, -1)
                                                                       if i % 2 == 1]
        # shuffle cycle
        if trng_seed is not None or trng_seed==0:
            rng = np.random.default_rng(trng_seed)
            rng.shuffle(cycle)
        medoids = medoids.loc[cycle[:]]
        print("\n\n***List of medoids***")
        print("---------------------")
        print(medoids.to_string())

        ### select the days for the validation set and the test set
        # medoid weights
        medoids["Weight"] = medoids[["Nb Samples"]] / medoids[
            ["Nb Samples"]].sum()  # np.ones(medoids.shape[0]) / medoids.shape[0]
        # numpy generator with seed
        rng = np.random.default_rng(val_test_seed)
        val_dates = pd.Series(index=medoids.index, dtype=object)
        test_dates = pd.Series(index=medoids.index, dtype=object)
        for medoid_name in medoids.sort_index().index:
            medoid_date = medoids.loc[medoid_name, "Date"]
            # # get the label of the medoid
            # medoid_label = medoid_labels.loc[medoid_date, "Cluster"]
            # get all the days associated with the medoid
            medoid_days = medoid_labels[medoid_labels["Cluster"] == medoid_name].index[2:-2]
            # remove the day used as medoid
            medoid_days = medoid_days.drop(medoid_date)
            # no dates between 02/29 and 03/05 to avoid leap year problems
            medoid_days = medoid_days[~((medoid_days.month == 2) & (medoid_days.day >= 29)) &
                                      ~((medoid_days.month == 3) & (medoid_days.day <= 5))]
            # sample one of the day associated with the medoid for the validation
            val_date = rng.choice(medoid_days)
            # remove the day used by validation
            medoid_days = medoid_days.drop(val_date)
            # sample one of the day associated with the medoid for the test
            test_date = rng.choice(medoid_days)
            val_dates[medoid_name] = pd.Timestamp(val_date)
            test_dates[medoid_name] = pd.Timestamp(test_date)

        return medoids, val_dates, test_dates


    def train(self, initial_model_folderpath: str, warm_start: str, nb_epochs_max: int, loss_metric: str, lr: float,
              gamma: float, update_freq: int, hyperparameters: dict,
              thermal_model: str, ems_relaxation_for_convex_opti: str, patience_max: int,
              seed: int, snr: float, std_w: float = 0, std_b: float = 0, std_requires_grad: bool = False, s: int = 1):

        # create validations and test datasets, which must be comparable across trainings (hence fixed seed)
        medoids, val_dates, test_dates = self.create_datasets(trng_seed=seed, val_test_seed=179)
        ### prepare training loop
        self.paths = paths_dic(initial_model_folderpath)
        weights_df, biases_df, losses_dic, ems_dic = logs(medoids.index, range(nb_epochs_max+1))

        now = pd.Timestamp.now().strftime("%Y-%m-%d_%Hh%Mm%Ss%f")
        init_flag = True
        days_cnt = 0  # count the number of training samples
        stopwatch_dic = {"training": Stopwatch(), "validation": Stopwatch(), "test": Stopwatch(),
                         "opti_training": Stopwatch(), "opti_validation": Stopwatch(), "opti_test": Stopwatch(),
                         "simu_training": Stopwatch(), "simu_validation": Stopwatch(), "simu_test": Stopwatch()}
        pending_losses = [] # list of the losses of that have been computed but not yet used for a gradient step

        # For each epoch (= representative year)
        for epoch in range(nb_epochs_max):
            # For each weather scenario (medoid) and each thermodynamics model (only bigM ReLU model for now)
            for medoid_name, relu_formulation in product(medoids.index, ["bigM"]):
                stopwatch_dic["training"].start()
                for i_s in range(s):  # the number of samples for stochastic smoothing (should be 1 for other relaxations)
                    # get the date of the medoid
                    day_date = medoids.loc[medoid_name, "Date"]
                    medoid_weight = medoids.loc[medoid_name, "Weight"]
                    print(f"\n\n-----------Epoch {epoch}/{nb_epochs_max - 1}, Medoid {medoid_name}, {day_date}----------")
                    ### Create the miqp ems (irrespective of the relaxation because it is used for validation and test)
                    if init_flag:
                        # build the true ems (ems_miqp) and get the initial weights and biases
                        self.ems_miqp, weights, biases, self.paths["cvxpylayer_model_save_folderpath"] = EMS.build_ems(
                            day_date, thermal_model, self.paths, now,None, "miqp", warm_start,
                            hyperparameters, snr)
                        if ems_relaxation_for_convex_opti == "qp":
                            # build the qp ems (ems_qp) to initialize the cvxpylayer
                            self.ems_qp, _, _, _ = EMS.build_ems(day_date, thermal_model, self.paths, now,
                                                                None, "qp", warm_start,
                                                                hyperparameters, snr)
                        # get the paths
                        self.paths["cvxpy_results_filepath"] = os.path.join(self.paths["cvxpylayer_model_save_folderpath"],
                                                                            "ResultsOpti.xlsx")
                        # get the weights and biases from the loaded nn or rc model
                        self.set_param_to_optimize((weights, biases))
                        # set the initial standard deviation of the parameters to optimize
                        self.set_param_to_optimize_std(std_w, std_b, std_requires_grad)
                    # record the weights and biases used for the sample
                    weights_df.at[epoch, medoid_name], biases_df.at[epoch, medoid_name] = self.get_param_to_optimize(requires_grad=False)
                    # run the ems
                    ems, (opti_time, simu_time) = self.run_ems(epoch, medoid_name, ems_dic, "training",
                                                               day_date, thermal_model, ems_relaxation_for_convex_opti)
                    stopwatch_dic["opti_training"].add_elapsed_time(opti_time)
                    stopwatch_dic["simu_training"].add_elapsed_time(simu_time)

                    if not ems.feasible:
                        print("Infeasible sample")
                        continue

                    if ems_relaxation_for_convex_opti in ["qp", "fixed_bin"]:  # no cvxpylayer needed for stochastic miqp
                        # if init, build the cvxpylayer
                        if init_flag:
                            self.build_cvxpylayer(ems, thermal_model, ems_relaxation_for_convex_opti)

                        # get the parameters for the fixed_bin cvxpy from the relaxed ems
                        cvxpylayer_parameters, _ = ems.get_cvxpylayer_parameters(
                            self.cvxpylayer_opti_formulation, ems_relaxation_for_convex_opti)
                        # get the parameters for the cvxpylayer
                        for k, v in self.cvxpylayer_param_to_optimize.items():
                            cvxpylayer_parameters[k] = v
                        # solve the cvxpylayer
                        warnings.filterwarnings("error")  # catch if the solver fails
                        try:
                            stopwatch_dic["opti_training"].start()
                            self.cvxpylayer_solution = self.cvxpylayer(*cvxpylayer_parameters.values(),
                                                                       solver_args={"solve_method": "ECOS", "verbose": False,
                                                                                    "max_iters": 2000000})
                            stopwatch_dic["opti_training"].stop()
                            warnings.resetwarnings()
                        except Exception as e:
                            print(f"Infeasible Sample: {e}")
                            # we were expecting a feasible sample (with Gurobi) but ECOS failed
                            ems_dic["training"].at[epoch, medoid_name].feasible = False
                            warnings.resetwarnings()
                            continue

                        solution = dict(zip(self.cvxpylayer_variable_names, self.cvxpylayer_solution))
                    else:  # for smiqp
                        solution = self.ems_miqp.cvxpy_solution

                    # compute the loss
                    pending_losses.append(get_losses(medoid_weight, loss_metric, ems, solution))
                # END of the for-loop: for each sample s

                # count the number of days used for training
                days_cnt += 1

                # check if at least one of the S samples is feasible
                if len(pending_losses) == 0:
                    print(f"Epoch {epoch} - Medoid {medoid_name} - No feasible sample")
                    continue

                # record the (detached) losses
                keys = pending_losses[0].keys()
                losses_dic["training"].at[epoch, medoid_name] = {k: torch.mean(torch.stack([d[k].detach().clone() for d in pending_losses]))
                                                                 for k in keys}

                ### Update the parameters
                # create the optimizer and learning rate scheduler at the end of the very first iteration
                if init_flag:
                    # Adam optimizer
                    self.optimizer = torch.optim.Adam(self.get_param_to_optimize_for_optimizer(), lr=lr)
                    # SGD optimizer
                    # self.optimizer = torch.optim.SGD(self.get_param_to_optimize_for_optimizer(), lr=lr)
                    # learning rate scheduler 'exponential decay'
                    self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=gamma)
                    init_flag = False

                # make one step of the gradient descent
                if days_cnt % update_freq == 0:
                    batch_loss = torch.stack([l["training_loss"] for l in pending_losses]).mean()
                    pending_losses = []
                    if ems_relaxation_for_convex_opti == "ss":
                        self.step_ss(batch_loss)
                    else:
                        self.step(batch_loss)
                    print(f'Training Loss: {batch_loss:.2f}')

                # stop counting time for training
                stopwatch_dic["training"].stop()

                # save the training optimization results
                save_results(self.paths["cvxpy_results_filepath"], medoid_name, relu_formulation, ems)
                # plot the scheduling results
                ncols = 5
                fig_path = os.path.join(self.paths["cvxpylayer_model_save_folderpath"], medoid_name)
                c.createdir(fig_path, up=0)
                # plot only if the last ems (i_s = S) is feasible
                if ems.feasible:
                    ems.plot_results(ncols, format=f"pdf", path=fig_path, savefig=True, showfig=False)
            # END of the for-loop: for each medoid

            # Check at least one sample in the last epoch is feasible
            any_fsbl = any_sample_feasible(ems_dic["training"].loc[epoch])
            if not any_fsbl:  # if no feasible sample: training is stopped
                print(f"Epoch {epoch} - No feasible sample")
                return

            # update the learning rate at each epoch
            self.scheduler.step()

            ### VALIDATION
            stopwatch_dic["validation"].start()
            # default value is validation sample is infeasible
            with (torch.no_grad()):
                for medoid_name in medoids.index:
                    val_date = val_dates[medoid_name]
                    print(f"\n\n-----------Validation: Epoch {epoch}/{nb_epochs_max - 1}, Medoid {medoid_name},"
                          f"{val_date}----------")
                    medoid_weight = medoids.loc[medoid_name, "Weight"]
                    # build the ems run for the validation
                    # if more than 10 neurons, and relaxation is qp, then use the qp relaxation
                    if (thermal_model == "nn"
                        and hyperparameters["nb_layers"] * hyperparameters["nb_neurons"] >= 10
                        and ems_relaxation_for_convex_opti == "qp"):
                        ems_val, (opti_time, simu_tim) = self.run_ems(epoch, medoid_name, ems_dic, "validation",
                                                                      val_date, thermal_model, "qp")
                    else:
                        ems_val, (opti_time, simu_tim) = self.run_ems(epoch, medoid_name, ems_dic, "validation",
                                                                      val_date, thermal_model, "miqp")
                    stopwatch_dic["opti_validation"].add_elapsed_time(opti_time)
                    stopwatch_dic["simu_validation"].add_elapsed_time(simu_tim)

                    if ems_val.feasible:
                        l_val = get_losses(medoid_weight, loss_metric, ems_val, ems_val.cvxpy_solution)
                        losses_dic["validation"].at[epoch, medoid_name] = c.map_dict(l_val, lambda x: x.detach())

            stopwatch_dic["validation"].stop()
            mean_val_losses_np = epoch_plots(ems_dic, losses_dic, loss_metric, self.paths)
            ems_dic = epoch_save(self.paths, ems_dic, losses_dic, weights_df, biases_df)

            # Early stopping
            if mean_val_losses_np[-1] == np.min(mean_val_losses_np):
                patience = 0
                best_epoch = epoch
            else:
                patience += 1
                if patience == patience_max:
                    break

            # Save the configuration of this run in the summary file (only after the first epoch)
            if epoch == 0:
                summary_save(self, hyperparameters, ems_dic, losses_dic, best_epoch, now, stopwatch_dic, seed, snr,
                             warm_start, ems_relaxation_for_convex_opti, epoch, nb_epochs_max, loss_metric, update_freq,
                             std_w, std_b, std_requires_grad, s)

            print_losses(epoch, nb_epochs_max, losses_dic, loss_metric)
        # END of the for-loop: for each epoch

        # record the last update of the weights and biases
        weights_df.iat[epoch + 1, 0], biases_df.iat[epoch + 1, 0] = self.get_param_to_optimize(requires_grad=False)

        ### TEST
        print("\n\nComputing test metrics")
        stopwatch_dic["test"].start()
        # load the best weights and biases associated with the best epoch
        self.set_param_to_optimize((weights_df.iat[best_epoch + 1, 0], biases_df.iat[best_epoch + 1, 0]))

        with torch.no_grad():
            for medoid_name in medoids.index:
                test_date = test_dates[medoid_name]
                print(f"\n\n-----------Test: Medoid {medoid_name}, {test_date}----------")
                medoid_weight = medoids.loc[medoid_name, "Weight"]
                # run the ems for the test
                ems_test, (opti_time, simu_time) = self.run_ems(epoch, medoid_name, ems_dic, "test",
                                                                      test_date, thermal_model, "miqp")
                stopwatch_dic["opti_test"].add_elapsed_time(opti_time)
                stopwatch_dic["simu_test"].add_elapsed_time(simu_time)
                if ems_test.feasible:
                    l_test = get_losses(medoid_weight, loss_metric, ems_test, ems_test.cvxpy_solution)
                    losses_dic["test"].at[0, medoid_name] = c.map_dict(l_test, lambda x: x.detach())
        stopwatch_dic["test"].stop()
        print(f"Average Test Loss: {losses_dic['test'].iloc[0].apply(lambda x: x[loss_metric]).mean():.2f}")

        ems_dic = epoch_save(self.paths, ems_dic, losses_dic, weights_df, biases_df)
        summary_save(self, hyperparameters, ems_dic, losses_dic, best_epoch, now, stopwatch_dic, seed, snr, warm_start,
                     ems_relaxation_for_convex_opti, epoch, nb_epochs_max, loss_metric, update_freq, std_w, std_b,
                     std_requires_grad, s)
        save_stopwatches(self.paths["cvxpylayer_model_save_folderpath"], stopwatch_dic)


    def test_model(self, initial_model_folderpath: str, warm_start: str, loss_metric: str,
                 hyperparameters: dict, thermal_model: str, seed: int, ems_relaxation_for_convex_opti: str):
        """
        Test Identify-then-Optimize (ITO) models.
        Args:
            initial_model_folderpath:
            loss_metric:
            medoids:
            medoid_labels:
            hyperparameters:
            thermal_model:
            seed:
            ems_relaxation_for_convex_opti:

        Returns:

        """
        medoids, _, test_dates = self.create_datasets(trng_seed=seed, val_test_seed=179)
        self.paths = paths_dic(initial_model_folderpath)
        weights_df, biases_df, losses_dic, ems_dic = logs(medoids.index, [0])

        stopwatch_dic = {"training": Stopwatch(), "validation": Stopwatch(), "test": Stopwatch(),
                         "opti_training": Stopwatch(), "opti_validation": Stopwatch(), "opti_test": Stopwatch(),
                         "simu_training": Stopwatch(), "simu_validation": Stopwatch(), "simu_test": Stopwatch()}
        snr = 0
        std_w, std_b = 0, 0
        std_requires_grad = False
        s = 0
        now = pd.Timestamp.now().strftime("%Y-%m-%d_%Hh%Mm%Ss%f")
        init_flag = True
        epoch = 0
        nb_epochs_max = 1
        best_epoch = 0
        update_freq = np.nan

        stopwatch_dic["test"].start()
        with torch.no_grad():
            for medoid_name in medoids.index:
                print(f"\n\n-----------Test: Medoid {medoid_name}, {medoids.loc[medoid_name, 'Date']}----------")
                test_date = test_dates[medoid_name]
                medoid_weight = medoids.loc[medoid_name, "Weight"]
                if init_flag:
                    # build the true ems (ems_miqp) and get the initial weights and biases
                    self.ems_miqp, weights, biases, self.paths["cvxpylayer_model_save_folderpath"] = EMS.build_ems(
                        test_date, thermal_model, self.paths, now, None, "miqp",
                        warm_start, hyperparameters, snr)
                    # get the paths
                    self.paths["cvxpy_results_filepath"] = os.path.join(self.paths["cvxpylayer_model_save_folderpath"],
                                                                        "ResultsOpti.xlsx")
                    # get the weights and biases from the loaded nn or rc model
                    self.set_param_to_optimize((weights, biases))
                # run the ems
                ems_test, (opti_time, simu_time) = self.run_ems(epoch, medoid_name, ems_dic, "test",
                                                                      test_date, thermal_model, "miqp")
                stopwatch_dic["opti_test"].add_elapsed_time(opti_time)
                stopwatch_dic["simu_test"].add_elapsed_time(simu_time)
                if ems_test.feasible:
                    l_test = get_losses(medoid_weight, loss_metric, ems_test, ems_test.cvxpy_solution)
                    losses_dic["test"].at[0, medoid_name] = c.map_dict(l_test, lambda x: x.detach())
        stopwatch_dic["test"].stop()
        print(f"Average Test Loss: {losses_dic['test'].iloc[0].apply(lambda x: x[loss_metric]).mean():.2f}")

        ems_dic = epoch_save(self.paths, ems_dic, losses_dic, weights_df, biases_df)
        summary_save(self, hyperparameters, ems_dic, losses_dic, best_epoch, now, stopwatch_dic, seed, snr, warm_start,
                     ems_relaxation_for_convex_opti, epoch, nb_epochs_max, loss_metric, update_freq, std_w, std_b,
                     std_requires_grad, s)
        save_stopwatches(self.paths["cvxpylayer_model_save_folderpath"], stopwatch_dic)






