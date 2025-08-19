"""
Author: Pietro Favaro
Date: April 28, 2024
Description: This file contains the implementation of a Declarative Neural Network (DNN) class.
        At the moment, the DNN is made of one cvxpy layer.
"""

from cvxpylayers.torch import CvxpyLayer
from itertools import product
from src.library.DeclarativeNNcommon import *
from src.library.Classes import Stopwatch
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
            return self._param_to_optimize
        else:
            return self.map_param_to_optimize(lambda x: x.requires_grad_(False))

    def get_param_to_optimize_for_optimizer(self):
        t_l = []
        for d in self.param_to_optimize:
            c.iter_dic(d, c.append_and_return, t_l, inplace=True)
        return t_l

    def map_param_to_optimize(self, f):
        """
        Apply a function to the leaf values param_to_optimize attribute.
        Args:
            f:

        Returns:

        """
        cp = deepcopy(self.param_to_optimize)
        for p in cp:  # p is a dictionary (building name -> layer name -> weights/biases)
            c.iter_dic(p, f)
        return cp

    def set_param_to_optimize(self, parameters):
        """
        Setter for the param_to_optimize attribute.
        :param parameters: tuple (weights, biases) containing dictionaries (building name -> layer name -> weights/biases)
        """
        self.param_to_optimize = parameters
        # the parameters are saved with gradient tracking per default
        self.param_to_optimize = self.map_param_to_optimize(lambda x: x.requires_grad_())

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

    def run_ems(self, init_flag, epoch, medoid_name, ems_dic, dataset_str, day_date, thermal_model, now,
                ems_relaxation, warm_start, hyperparameters, snr, std_w=0, std_b=0):
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
            if std_w != 0 or std_b != 0:
                raise ValueError(f"std_w and std_b must be 0 for validation and test.")

        ### SOLVE THE TRUE MIQP
        if ems_relaxation == "fixed_bin" or ems_relaxation == "miqp" or ems_relaxation == "ss":
            if not init_flag:
                # rebuild the true ems (ems_miqp) with the updated parameters
                self.ems_miqp, _, _, _ = build_ems(day_date, thermal_model, self.paths, now,
                                                  self.ems_miqp.cvxpy_opti_formulation, "miqp",
                                                  warm_start, hyperparameters, snr)

            # Apply noise to the parameters
            if ems_relaxation == "ss":
                self.smiqp_param_sd = (wrapper_normal_distribution(self.param_to_optimize[0], std_w, self.ems_miqp),
                                      wrapper_normal_distribution(self.param_to_optimize[1], std_b, self.ems_miqp))
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
                solve_ems(self.ems_miqp, {'verbose': True, 'TimeLimit': 1200, 'MIPGap': 0.01, 'Threads': 0})
                ems_dic[dataset_str].at[0, medoid_name] = self.ems_miqp
            else:
                solve_ems(self.ems_miqp, {'verbose': True, 'TimeLimit': 60, 'MIPGap': 0.01, 'Threads': 0})
                ems_dic[dataset_str].at[epoch, medoid_name] = self.ems_miqp
            return self.ems_miqp
        ### SOLVE THE QP RELAXATION FOR CONVEXITY
        elif ems_relaxation == "qp":
            cvxpy_opti_formulation = None if init_flag else self.ems_qp.cvxpy_opti_formulation
            self.ems_qp, _, _, _ = build_ems(day_date, thermal_model, self.paths, now,
                                            cvxpy_opti_formulation, "qp",
                                            warm_start, hyperparameters, snr)
            # Set the ems parameters
            self.ems_qp.set_cvxpy_parameters(*self.param_to_optimize, nd_load_flag=False)
            # solve the ems (even though it is not necessary in this case but it simplifies the code)
            solve_ems(self.ems_qp, {'verbose': True, 'TimeLimit': 60, 'Threads': 0})

            # record the ems
            ems_dic[dataset_str].at[epoch, medoid_name] = self.ems_qp
            return self.ems_qp
        else:
            raise ValueError(f"Unknown relaxation for the ems optimization: {ems_relaxation}."
                             f" Only 'fixed_bin', 'miqp', and 'qp' are supported.")


    def create_datasets(self, medoids, medoid_labels, seed: int):
        """
        Args:
            medoids:
                DataFrame with the medoids and their weights
            medoid_labels:
                DataFrame with the labels of the medoids
            seed:
                Seed for the random number generator

        Returns:
            val_dates:
                Series with the validation dates for each medoid
            test_dates:
                Series with the test dates for each medoid
        """
        ### select the days for the validation set and the test set
        # medoid weights
        medoids["Weight"] = medoids[["Nb Samples"]] / medoids[
            ["Nb Samples"]].sum()  # np.ones(medoids.shape[0]) / medoids.shape[0]
        # numpy generator with seed
        rng = np.random.default_rng(seed)
        val_dates = pd.Series(index=medoids.index, dtype=object)
        test_dates = pd.Series(index=medoids.index, dtype=object)
        for medoid_name in medoids.index:
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

        return val_dates, test_dates


    def train(self, warm_start: str, nb_epochs_max: int, loss_metric: str, lr: float, gamma: float, update_freq: int,
              medoids, medoid_labels, hyperparameters: dict,
              thermal_model: str, ems_relaxation_for_convex_opti: str, patience_max: int,
              seed: int, snr: float, std_w: float = 0, std_b: float = 0):

        val_dates, test_dates = self.create_datasets(medoids, medoid_labels, seed)
        ### prepare training loop
        self.paths = paths_dic()
        weights_df, biases_df, losses_dic, ems_dic = logs(medoids.index, range(nb_epochs_max+1))

        now = pd.Timestamp.now().strftime("%Y-%m-%d_%Hh%Mm%Ss%f")
        init_flag = True
        sample_cnt = 0  # count the number of training samples
        stopwatch_dic = {"training": Stopwatch(), "validation": Stopwatch(), "test": Stopwatch()}

        # For each epoch (= representative year)
        for epoch in range(nb_epochs_max):
            # For each weather scenario (medoid) and each thermodynamics model (only bigM ReLU model for now)
            for medoid_name, relu_formulation in product(medoids.index, ["bigM"]):
                stopwatch_dic["training"].start()
                # get the date of the medoid
                day_date = medoids.loc[medoid_name, "Date"]
                medoid_weight = medoids.loc[medoid_name, "Weight"]
                print(f"\n\n-----------Epoch {epoch}/{nb_epochs_max - 1}, Medoid {medoid_name}, {day_date}----------")
                ### Create the miqp ems (irrespective of the relaxation because it is used for validation and test)
                if init_flag:
                    # build the true ems (ems_miqp) and get the initial weights and biases
                    self.ems_miqp, weights, biases, self.paths["cvxpylayer_model_save_folderpath"] = build_ems(
                        day_date, thermal_model, self.paths, now,None, "miqp",
                                                               warm_start, hyperparameters, snr)
                    # get the paths
                    self.paths["cvxpy_results_filepath"] = os.path.join(self.paths["cvxpylayer_model_save_folderpath"],
                                                                        "ResultsOpti.xlsx")
                    # get the weights and biases from the loaded nn or rc model
                    self.set_param_to_optimize((weights, biases))
                # record the weights and biases used for the sample
                weights_df.at[epoch, medoid_name], biases_df.at[epoch, medoid_name] = self.get_param_to_optimize(requires_grad=False)
                # run the ems
                ems = self.run_ems(init_flag, epoch, medoid_name, ems_dic, "training", day_date, thermal_model, now,
                               ems_relaxation_for_convex_opti, warm_start, hyperparameters, snr, std_w, std_b)

                if not ems.feasible:
                    print("Infeasible sample")
                    continue

                if ems_relaxation_for_convex_opti in ["qp", "fixed_bin"]:  # no cvxpylayer needed for stochastic miqp
                    # if init, build the cvxpylayer
                    if init_flag:
                        self.build_cvxpylayer(ems, thermal_model, ems_relaxation_for_convex_opti)

                    # get the parameters for the fixed_bin cvxpy from the miqp ems
                    cvxpylayer_parameters, _ = ems.get_cvxpylayer_parameters(
                        self.cvxpylayer_opti_formulation, ems_relaxation_for_convex_opti)
                    # get the parameters for the cvxpylayer
                    for k, v in self.cvxpylayer_param_to_optimize.items():
                        cvxpylayer_parameters[k] = v
                    # solve the cvxpylayer
                    try:
                        self.cvxpylayer_solution = self.cvxpylayer(*cvxpylayer_parameters.values(),
                                                                   solver_args={"solve_method": "ECOS", "verbose": False,
                                                                                "max_iters": 2000000})
                    except Exception as e:
                        print(f"{e}")
                        ems.feasible = False
                        continue

                    solution = dict(zip(self.cvxpylayer_variable_names, self.cvxpylayer_solution))
                else:  # for smiqp
                    solution = self.ems_miqp.cvxpy_solution

                ### compute the loss
                l = get_losses(medoid_weight, loss_metric, ems, solution)
                # record the (detached) losses
                losses_dic["training"].at[epoch, medoid_name] = c.map_dict(l, lambda x: x.detach())

                ### Update the parameters
                # create the optimizer and learning rate scheduler at the end of the very first iteration
                if init_flag:
                    # Adam optimizer
                    self.optimizer = torch.optim.Adam(self.get_param_to_optimize_for_optimizer(), lr=lr)
                    # SGD optimizer
                    # self.optimizer = torch.optim.SGD(self.get_param_to_optimize_for_optimizer(), lr=lr)
                    # learning rate scheduler 'exponential decay'
                    self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=gamma)

                # make one step of the gradient descent
                if sample_cnt % update_freq == 0:
                    if ems_relaxation_for_convex_opti == "ss":
                        self.step_ss(l["training_loss"])
                    else:
                        self.step(l["training_loss"])

                # stop counting time for training
                stopwatch_dic["training"].stop()

                # save the training optimization results
                save_results(self.paths["cvxpy_results_filepath"], medoid_name, relu_formulation, ems)
                # plot the scheduling results
                ncols = 5
                fig_path = os.path.join(self.paths["cvxpylayer_model_save_folderpath"], medoid_name)
                c.createdir(fig_path, up=0)
                ems.plot_results(ncols, format=f".pdf", path=fig_path, savefig=True, showfig=False)

                print(f'Training Loss: {l[loss_metric]:.2f}')
                init_flag = False
            # END of the for-loop: for each medoid

            # Check at least one sample in the last epoch is feasible
            any_fsbl = any_sample_feasible(ems_dic["training"].loc[epoch])
            if not any_fsbl:  # if no feasible sample: training is stopped
                print(f"Epoch {epoch} - No feasible sample")
                return

            # update the learning rate
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
                    # if 1 layer with 10 neurons or more, and relaxation is qp, then use the qp relaxation
                    if thermal_model == "nn" and hyperparameters["nb_layers"] >= 1 \
                            and hyperparameters["nb_neurons"] >= 10 and ems_relaxation_for_convex_opti == "qp":
                        ems_val = self.run_ems(init_flag, epoch, medoid_name, ems_dic, "validation",
                                               val_date, thermal_model, now, "qp", warm_start,
                                               hyperparameters, snr)
                    else:
                        ems_val = self.run_ems(init_flag, epoch, medoid_name, ems_dic, "validation",
                                                    val_date, thermal_model, now, "miqp", warm_start,
                                                    hyperparameters, snr)
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
                             std_w, std_b)

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
                # build the ems run for the validation
                ems_test = self.run_ems(init_flag, epoch, medoid_name, ems_dic, "test",
                                   test_date, thermal_model, now, "miqp", warm_start,
                                   hyperparameters, snr)
                if ems_test.feasible:
                    l_test = get_losses(medoid_weight, loss_metric, ems_test, ems_test.cvxpy_solution)
                    losses_dic["test"].at[0, medoid_name] = c.map_dict(l_test, lambda x: x.detach())
        stopwatch_dic["test"].stop()
        print(f"Average Test Loss: {losses_dic['test'].iloc[0].apply(lambda x: x[loss_metric]).mean():.2f}")

        ems_dic = epoch_save(self.paths, ems_dic, losses_dic, weights_df, biases_df)
        summary_save(self, hyperparameters, ems_dic, losses_dic, best_epoch, now, stopwatch_dic, seed, snr, warm_start,
                     ems_relaxation_for_convex_opti, epoch, nb_epochs_max, loss_metric, update_freq, std_w, std_b)

        final_brkpt = 1

    def test_ito(self, loss_metric, medoids, medoid_labels, hyperparameters: dict, thermal_model: str, seed: int):
        """
        Test Identify-then-Optimize (ITO) models.
        Args:
            loss_metric:
            medoids:
            medoid_labels:
            hyperparameters:
            thermal_model:
            seed:

        Returns:

        """
        _, test_dates = self.create_datasets(medoids, medoid_labels, seed)
        self.paths = paths_dic()
        weights_df, biases_df, losses_dic, ems_dic = logs(medoids.index, [0])

        stopwatch_dic = {"training": Stopwatch(), "validation": Stopwatch(), "test": Stopwatch()}
        snr = 0
        std_w, std_b = 0, 0
        now = pd.Timestamp.now().strftime("%Y-%m-%d_%Hh%Mm%Ss%f")
        warm_start = "True"
        init_flag = True
        epoch = 0
        nb_epochs_max = 1
        best_epoch = 0
        update_freq = np.nan
        ems_relaxation_for_convex_opti = ""

        stopwatch_dic["test"].start()
        with torch.no_grad():
            for medoid_name in medoids.index:
                print(f"\n\n-----------Test: Medoid {medoid_name}, {medoids.loc[medoid_name, 'Date']}----------")
                test_date = test_dates[medoid_name]
                medoid_weight = medoids.loc[medoid_name, "Weight"]
                if init_flag:
                    # build the true ems (ems_miqp) and get the initial weights and biases
                    self.ems_miqp, weights, biases, self.paths["cvxpylayer_model_save_folderpath"] = build_ems(
                        test_date, thermal_model, self.paths, now, None, "miqp",
                        warm_start, hyperparameters, snr)
                    # get the paths
                    self.paths["cvxpy_results_filepath"] = os.path.join(self.paths["cvxpylayer_model_save_folderpath"],
                                                                        "ResultsOpti.xlsx")
                    # get the weights and biases from the loaded nn or rc model
                    self.set_param_to_optimize((weights, biases))
                # build the ems run for the validation
                ems_test = self.run_ems(init_flag, epoch, medoid_name, ems_dic, "test",
                                   test_date, thermal_model, now, "miqp", warm_start,
                                   hyperparameters, snr)
                if ems_test.feasible:
                    l_test = get_losses(medoid_weight, loss_metric, ems_test, ems_test.cvxpy_solution)
                    losses_dic["test"].at[0, medoid_name] = c.map_dict(l_test, lambda x: x.detach())
        stopwatch_dic["test"].stop()
        print(f"Average Test Loss: {losses_dic['test'].iloc[0].apply(lambda x: x[loss_metric]).mean():.2f}")

        ems_dic = epoch_save(self.paths, ems_dic, losses_dic, weights_df, biases_df)
        summary_save(self, hyperparameters, ems_dic, losses_dic, best_epoch, now, stopwatch_dic, seed, snr, warm_start,
                     ems_relaxation_for_convex_opti, epoch, nb_epochs_max, loss_metric, update_freq, std_w, std_b)






