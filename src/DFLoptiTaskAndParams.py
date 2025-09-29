"""
Create the list of parameters to be executed in parallel for the DFL optimization.
"""
import dill as pickle
from itertools import product, repeat
from joblib import Parallel, delayed
import math
import numpy as np
import os
import pandas as pd
import traceback
import warnings
import src.library.Common as c
from src.library.DeclarativeNN import DeclarativeNN


####################################
### Parameters
####################################
# N.B., the file 18zones_ASHRAE901_OfficeMedium_STD2019_Denver.idf does not account for daylight saving time.
# Also, it is recommended to perform all the operation in UTC time and convert the results to the local time zone
# only for visualization purposes.


def get_initial_model_folderpath(param):
    """
    given a set of parameters, return the filepath of the initial thermal dynamics model to load
    Args:
        param: a dictionary containing the parameters for the training (param is an elemnt from params_all)
    Returns:
        filepath: the filepath of the initial thermal dynamics model to load
    """
    # The initial model is a warm-start (no DFL training) it is not a test_only (DFL training)
    # or it evaluates the warm-start model
    thermal_model = param["thermal_model"]
    hyperparameters = param["hpp"]
    # if there is a non-dfl (i.e., supervized learning) warm start model to load
    if not param["test_only"] or (param["test_only"] and param["warm_start"] == "True"):
        if thermal_model == "nn":
            # load the nn model
            nn_summary_filepath = os.path.abspath("data/SmallOffice/Models/NN/TrainingSummary_NN2.xlsx")
            NN_datetime = c.getbestmodel(nn_summary_filepath, hyperparameters)
            NN_tail_filepath = (f"NotSparse/NN_{hyperparameters['nb_layers']}layers_"
                                f"{hyperparameters['nb_neurons']}neurons/{NN_datetime}")
            initial_model_folderpath = os.path.join(os.path.dirname(nn_summary_filepath), NN_tail_filepath)
        elif thermal_model == "rcmodel":
            # load the rc model
            rc_summary_filepath = os.path.abspath("data/SmallOffice/Models/RC/summary_RC.xlsx")
            rc_datetime = c.getbestmodel(rc_summary_filepath, hyperparameters, metric="MAE_24h", sense="min")
            initial_model_folderpath = os.path.join(os.path.dirname(rc_summary_filepath), rc_datetime)
        elif thermal_model == "spatialrcmodel":
            # load the rc model
            rc_summary_filepath = os.path.abspath("data/SmallOffice/Models/spatialRC/summary_spatialRC.xlsx")
            rc_datetime = c.getbestmodel(rc_summary_filepath, hyperparameters, metric="MAE_24h", sense="min")
            initial_model_folderpath = os.path.join(os.path.dirname(rc_summary_filepath), rc_datetime, "RCmodel.pth")
        else:
            raise ValueError("The thermal model must be either 'nn' or '(spatial)rcmodel'.")
    # to test a model already trained with dfl
    elif param["test_only"] and "dfl" in param["warm_start"]:
        # to test a dfl model, the initial model to load was not test_only and was not warm_start dfl
        modified_param = param.copy()
        modified_param["test_only"] = False
        modified_param["warm_start"] = "True"
        # path the summary file of the dfl trainings and model tests
        dfl_summary_filepath = os.path.abspath("data/SmallOffice/Models/cvxpylayer/summary.xlsx")
        # find the datetime of the model trained with dfl
        NN_datetimes = find_params_in_summary([modified_param], dfl_summary_filepath)[0]
        if NN_datetimes is None:
            raise ValueError(f"No initial model to load for test_only with warm_start={param['warm_start']}")
        elif len(NN_datetimes) > 1:
            raise ValueError(f"Multiple initial models to load for test_only with warm_start={param['warm_start']}\n", NN_datetimes)
        else:
            NN_datetime = NN_datetimes[0]

        if thermal_model == "nn":
            # tail filepath for nn model
            tail_filepath = (f"NotSparse/NN_{hyperparameters['nb_layers']}layers_"
                                f"{hyperparameters['nb_neurons']}neurons/{NN_datetime}")
        elif thermal_model == "rcmodel":
            # tail filepath for rc model
            tail_filepath = f"RC/{NN_datetime}"
        elif thermal_model == "spatialrcmodel":
            # tail filepath for spatial RC model
            tail_filepath = f"spatialRC/{NN_datetime}"
        else:
            raise ValueError("The thermal model must be either 'nn' or '(spatial)rcmodel'.")
        initial_model_folderpath = os.path.join(os.path.dirname(dfl_summary_filepath), tail_filepath)
    else:
        raise ValueError("No initial model to load for test_only with warm_start='False'")

    return initial_model_folderpath


def create_parameter_list():
    ems_relaxations = ["fixed_bin", "qp", "ss"]  # "fixed_bin", "qp", "ss"
    nb_epochs_max = 100
    seeds = range(0, 5)  # seed for reproducibility

    hyperparameters = [
        ("nn", {"target": "Zone Mean Air Temperature(t+1)", "nb_layers": 1, "nb_neurons": 2, "nb_inputs": 11,
                          "activation": "ReLU()", "sparse": False}),
        ("nn", {"target": "Zone Mean Air Temperature(t+1)", "nb_layers": 1, "nb_neurons": 5, "nb_inputs": 11,
                           "activation": "ReLU()", "sparse": False}),
        ("nn", {"target": "Zone Mean Air Temperature(t+1)", "nb_layers": 1, "nb_neurons": 10, "nb_inputs": 11,
                "activation": "ReLU()", "sparse": False}),
        ("rcmodel", {"target": "Zone Mean Air Temperature(t+1)"}),
    ]

    # Loss metric is given by the ems_relaxation: "qp" and "fixed_bin" -> "hierarchical_weighted_mae", "ss" -> "expost+"
    # loss_metrics = ["expost+", "hierarchical_weighted_mae"]
    learning_rates = [1e-3] # [1e-3, 5e-3, 1e-2]
    gamma = 0.98
    update_frequency = 1  # update the weights every x medoids
    # Whether to warm start the NN/RC training with the initial model (trained on historical data) or not
    # If warm_start is False and test_only is False, then the DFL model is trained from scratch
    # "Noise" means warm start with noise of amplitude signal-to-noise ratio = SNR on the weights
    warm_start = "Noise"  # "Noise" or "True" or "False"
    snr = 1000
    # value of the std and whether the std is a learnable parameter (if std init = cst) or not
    stds = [
        (0, False),
        (0.01, False),
        (0.05, False), (0.1, False),
            (0.01, True), (0.05, True), (0.1, True),
            (lambda x: x/2, False), (lambda x: x/10, False),
            (lambda x: x/2, True), (lambda x: x/10, True)
            ]
    S = [1]  # number of samples to approximate the gradient with stochastic smoothing
    test_only_l = [False]  # True, False

    params = []
    for test_only in test_only_l:
        for thermal_model, hpp in hyperparameters:
            # If test_only is True, we do not need to specify ems_relax, learning_rate, stds, and seed
            if test_only:
                params.append({
                    "thermal_model": thermal_model,
                    "hpp": hpp,
                    "loss_metric": "expost+",
                    "ems_relax": np.nan,  # no need to specify ems_relax for test_only
                    "learning_rate": np.nan,  # no need to specify ems_relax for test_only
                    "gamma": np.nan,
                    "update_frequency": np.nan,
                    "warm_start": warm_start,
                    "seed": 0,
                    "snr": np.nan,
                    "stds": np.nan,
                    "s": np.nan,
                    "test_only": test_only,
                    "nb_epochs_max": 1,
                })
                continue

            for seed in seeds:
                for ems_relax, learning_rate in product(ems_relaxations, learning_rates):
                    loss_metric = "expost+" if ems_relax == "ss" else "hierarchical_weighted_mae"

                    for (std, rg), s in product(stds, S):
                        params.append({
                            "thermal_model": thermal_model,
                            "hpp": hpp,
                            "loss_metric": loss_metric,
                            "ems_relax": ems_relax,
                            "learning_rate": learning_rate,
                            "gamma": gamma,
                            "update_frequency": update_frequency,
                            "warm_start": warm_start,
                            "seed": seed,
                            "snr": snr,
                            "stds": std,
                            "std_requires_grad": rg,
                            "s": s,
                            "test_only": test_only,
                            "nb_epochs_max": nb_epochs_max,
                        })

    # For each parameter set, get the initial model folder path
    for p in params:
        p["initial_model_folderpath"] = get_initial_model_folderpath(p)

    return params


def manual_modification_of_parameters(params):
    # change manually some parameters to improve performance
    for p in params:
        # for rcmodel and qp relaxation, we use a higher learning rate
        if not p["test_only"] and p["thermal_model"] == "rcmodel" and p["ems_relax"] == "qp":
            p["learning_rate"] = 2e-2
        # if test_only, the maximum number of epochs is 1
        if p["test_only"]:
            p["nb_epochs_max"] = 1
    return params

def split_params(params_all, n_tasks):
    """
    Splits params_all into n_tasks chunks, and save the parameters for the given task_id.
    Handles uneven division by distributing the remainder.
    task_id should be in the range [0, n_tasks-1].
    """
    n_params = len(params_all)
    # Compute base chunk size and remainder
    base = n_params // n_tasks
    rem = n_params % n_tasks

    # Check if save folder exists, if not create it
    folder_path = "output/task_parameters"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    else:
        # If the folder already exists, remove all the files in it
        for f in os.listdir(folder_path):
            os.remove(os.path.join(folder_path, f))

    def split_params_modulo(params_all, n_tasks):
        """
        Splits params_all into n_tasks lists based on index mod n_tasks.
        Each list i gets the params at indices where idx % n_tasks == i.
        Returns a list of lists, each for a task.
        """
        task_params = [[] for _ in range(n_tasks)]
        for idx, param in enumerate(params_all):
            task_id = idx % n_tasks
            task_params[task_id].append(param)
        return task_params

    # # Compute start and end indices for each task and save the parameters
    # for task_id in range(n_tasks):
    #     start = task_id * base + min(task_id, rem)
    #     end = start + base + (1 if task_id < rem else 0)
    #     task_params = params_all[start:end]

    task_params = split_params_modulo(params_all, n_tasks)
    for task_id in range(n_tasks):
        # print task parameters for debugging
        print(f"\n\n***Parameters for task {task_id}***")
        print('---------------------------------')
        print_parameter_sets(task_params[task_id])
        # Save the parameters for this task
        task_filepath = os.path.join("output/task_parameters", f"task_{task_id}_params.pkl")
        if not os.path.exists(os.path.dirname(task_filepath)):
            os.makedirs(os.path.dirname(task_filepath))
        with open(task_filepath, "wb") as f:
            pickle.dump(task_params[task_id], f)

############################################
### Process task
############################################
# Function to be run in parallel
def process_task(params):
    """

    Args:
        params: a dictionary containing the parameters for the training

    Returns:

    """
    thermal_model = params["thermal_model"]
    ems_relax = params["ems_relax"]
    hpp = params["hpp"]
    loss_metric = params["loss_metric"]
    seed = params["seed"]
    test_only = params["test_only"]
    nb_epochs_max = params["nb_epochs_max"]
    initial_model_folderpath = params["initial_model_folderpath"]
    warm_start = params["warm_start"]

    seed, _, _ = c.make_reproducible(seed)
    ### to test ITO models
    dnn = DeclarativeNN()
    if test_only or "dfl" in warm_start:
        dnn.test_model(initial_model_folderpath, warm_start, loss_metric, hpp, thermal_model,
                       seed, ems_relax)
        return
    ### to train
    # add the hyperparameters
    learning_rate = params["learning_rate"]
    gamma = params["gamma"]
    update_frequency = params["update_frequency"]
    snr = params["snr"]
    stds = params["stds"]
    std_requires_grad = params["std_requires_grad"]
    s = params["s"]
    if "rcmodel" in thermal_model and ems_relax == "fixed_bin":
        raise Warning("RC model must be used with QP relaxation or SS relaxation. This run configuration was skipped.")
    dnn.train(initial_model_folderpath, warm_start, nb_epochs_max, loss_metric, learning_rate, gamma, update_frequency,
              hpp, thermal_model, ems_relax,
              patience_max=15, seed=seed, snr=snr, std_w=stds, std_b=stds, std_requires_grad = std_requires_grad,
              s=s)


def safe_process_task(params):
    """
    Wrapper to handle exceptions during parallel processing.
    This function will catch any exception and print the parameters that caused it.
    """
    try:
        process_task(params)
    except Exception as e:
        print(f"Error processing parameters: {params}")
        print(traceback.format_exc())
        if not os.path.exists("output/error_log"):
            os.mkdir("output/error_log")
        now = pd.Timestamp.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")
        error_filepath = os.path.join("output/error_log", f"error_{now}.txt")
        with open(error_filepath, "a") as f:
            f.write(f"Error processing parameters: {params}\nException: {e}\n")

def print_parameter_sets(params):
    if len(params) == 0:
        print("No parameters to display.")
        return
    params_df = pd.DataFrame(params, columns=params[0].keys())
    d_hpp_df = pd.DataFrame(params_df['hpp'].tolist())
    # to avoid bug, make sure there is 'nb_layers' and 'nb_neurons' columns
    if 'nb_layers' not in d_hpp_df.columns or 'nb_neurons' not in d_hpp_df.columns:
        d_hpp_df = pd.concat([d_hpp_df, pd.DataFrame(columns=['nb_layers', 'nb_neurons'])], ignore_index=True)
    print(pd.concat([params_df["thermal_model"], d_hpp_df[['nb_layers', 'nb_neurons']],
                     params_df[["ems_relax", "stds", "s", "seed", "loss_metric", "learning_rate", "test_only"]]],
                    axis=1).to_string())


def decorator_summary_file_filter(filter_to_apply):
    """
    Return a function that reads the summary file (wrap_summary_file_filter)
    and filter (according to filter_to_apply) the parameters that have been already run successfully.
    """
    def wrap_summary_file_filter(params_all, summary_filepath):
        """
            read the summary file and filter the parameters that have been already run successfully.
        """
        try :
            # read the summary file
            summary_df = pd.read_excel(summary_filepath, index_col=0)
            # Keep only successful runs
            summary_df = summary_df.loc[list(map(lambda x: not math.isnan(x), summary_df["Test Loss"]))]
        except FileNotFoundError:
            # use a print rather than a warning because warning print is not displayed at the right place
            print("\033[31mWARNING: Summary file does not exist. No filtering with summary file will be applied.\033[0m")
            summary_df = pd.DataFrame()
        return filter_to_apply(params_all, summary_df)
    return wrap_summary_file_filter


@decorator_summary_file_filter
def summary_file_filter(params_all, summary_df):
    """
        goes through the runs in the summary file and filter the parameter sets that have been already run successfully.

        Args:
            params_all: list of parameter sets to filter
            summary_df: dataframe containing the summary of the runs
        Returns:
            a list of booleans indicating which parameters should be kept
    """
    if len(summary_df) == 0:
        return [True] * len(params_all)
    else:
        summary_diclist = summary_df.to_dict(orient="records")
        return [not any(list(map(match_summary_and_params, summary_diclist, repeat(p)))) for p in
                params_all]


@decorator_summary_file_filter
def find_params_in_summary(params_all, summary_df):
    """
    goes through the summary file of the runs and find the timestamp of the run associated with the parameter sets
    of params_all.

    Args:
        params_all: list of parameter sets to search for in the summary file
        summary_df: dataframe containing the summary of the runs
    Returns:
        a list with the timestamps associated to the successful matches
    """
    params_df = pd.DataFrame(params_all)
    if len(summary_df) == 0:
        return [None] * len(params_all)
    else:
        matches_ts = []
        summary_diclist = summary_df.to_dict(orient="records")
        # for each parameter set
        for ps in params_all:
            # match each parameter set against all the summary rows
            match_p = list(map(match_summary_and_params, summary_diclist, repeat(ps)))
            if any(match_p):
                matches_ts.append(list(summary_df.loc[match_p, "Date"].values))  # get the timestamp of all matches
            else:
                matches_ts.append(None)
        return matches_ts


def match_summary_and_params(summary_row, one_param_set):
    """
    Check if the summary row matches the parameter set.
    """
    match = []

    # check maximum number of epochs
    summary_nb_epochs, summary_max_epochs = map(int, summary_row["Nb Epoch"].split("/"))
    match.append(one_param_set["nb_epochs_max"] == summary_max_epochs)
    # check the test only flag
    summary_test_only_flag = (summary_max_epochs == 1 and summary_row["Training time"] == 0 and summary_row[
        "Validation time"] == 0)
    match.append(one_param_set["test_only"] == summary_test_only_flag)

    # checks only possible for training runs (no test_only)
    if not one_param_set["test_only"] and summary_test_only_flag is False:
        # Check if the relaxation matches
        match.append(one_param_set["ems_relax"] == summary_row["Bin. Formulation"])
        # Check the warm start
        match.append(one_param_set["warm_start"] == str(summary_row["Warm-start"]))
        # Check the stds
        match.append(
            str(one_param_set["stds"]) == summary_row["Std w"] and str(one_param_set["stds"]) == summary_row["Std b"])
        # Check if std_requires_grad matches
        match.append(one_param_set["std_requires_grad"] == summary_row["Std requires grad"])
        # Check the S
        match.append(one_param_set["s"] == summary_row["S"])
        # Check the learning rate
        summary_initial_lr = np.round(summary_row["Learning rate"] / (summary_row["gamma"] ** (summary_nb_epochs + 1)),
                                      decimals=8)
        match.append(one_param_set["learning_rate"] == summary_initial_lr)
        # Check the gamma
        match.append(one_param_set["gamma"] == summary_row["gamma"])
    # check the model characteristics
    tm = one_param_set["thermal_model"]
    if tm != "nn":
        match.append(tm == summary_row["Model carac."])
    else:
        # For nn, we need to check the hpp
        hpp = one_param_set["hpp"]
        mc = f"{hpp['nb_layers']}layers_{hpp['nb_neurons']}neurons_each"
        match.append(mc == summary_row["Model carac."])
    # Check the loss metric
    match.append(one_param_set["loss_metric"] == summary_row["Loss metric"])
    # Check the seed
    match.append(one_param_set["seed"] == summary_row["Seed"])

    return all(match)


def manual_filtering_of_parameters(p):
    """
    p is one element of params_all
    return: True if the parameter set should be kept, False otherwise
    """
    # remove all the "fixed_bin" and "ss" configurations for 1x10 NN model.
    if (not p["test_only"] and
            (p["thermal_model"] == "nn" and p["hpp"]["nb_neurons"] == 10 and p["ems_relax"] in ["fixed_bin", "ss"])):
        warnings.warn(f"To mitigate computational burden, 1x10 NN model must be used with QP relaxation,"
                      f"not {p['ems_relax']}.")
        return False
    # remove all the "ss" and "fixed_bin" configurations for RC model.
    if (not p["test_only"] and
            (p["thermal_model"] == "rcmodel" and p["ems_relax"] in ["fixed_bin", "ss"])):
        warnings.warn(f"RC model must be used with QP relaxation, not {p['ems_relax']}."
                      f"This run configuration was skipped.")
        return False
    # remove all the "ss" with std = 0 without gradient
    if (not p["test_only"] and
            (p["ems_relax"] == "ss" and p["stds"] == 0 and p["std_requires_grad"] is False)):
        warnings.warn(f"SS relaxation with std=0 and std_requires_grad=False is not a valid configuration.")
        return False
    # remove all the "qp" and "fixed_bin" with std != 0
    if (not p["test_only"] and
            (p["ems_relax"] in ["qp", "fixed_bin"] and p["stds"] != 0)):
        warnings.warn("QP and fixed_bin relaxations must be used with std=0.")
        return False
    # remove all the "qp" and "fixed_bin" with S != 1 (if not testing)
    if (not p["test_only"] and
            (p["ems_relax"] in ["qp", "fixed_bin"] and p["s"] != 1)):
        warnings.warn("QP and fixed_bin relaxations must be used with S=1.")
        return False
    return True



def filter_params(params_all, summary_filepath):
    """
    Apply some manual filtering to the parameters.
    Then, read the summary file and filter the parameters that have been already run successfully.
    Then, look into the error log folder and filter the parameters that have been run but failed.

    Args:
        params_all: list of parameters to filter
        summary_filepath: path to the summary file
    Returns:
        a list of booleans indicating which parameters should be kept
    """

    ### Manual filtering of parameters
    # remove all the "fixed_bin" and "ss" configurations for 1x10 NN model.
    filter_bool1 = [manual_filtering_of_parameters(p) for p in params_all]

    ### Automatic filtering based on the summary file
    filter_bool2 = summary_file_filter(params_all, summary_filepath)

    ### Remove the parameters that have been run but failed
    if not os.path.exists("output/error_log"):
        # use a print rather than a warning because warning print is not displayed at the right place
        print("\033[31mWARNING: Error log folder does not exist. No filtering with error log will be applied.\033[0m")
        filter_bool3 = [True] * len(params_all)
    else:
        # read all the error files in the error_log folder
        error_files = [f for f in os.listdir("output/error_log") if f.endswith(".txt")]
        filter_bool3 = []
        for p in params_all:
            # remove medoids and medoid_labels from the parameters because they are compressed in the error files
            # it forces to remove the nb_epochs_max parameter as well
            p = {k: v for k, v in p.items() if k not in ["medoids", "medoid_labels", "nb_epochs_max"]}
            # Check if the parameters are in the error files
            found = False
            for ef in error_files:
                with open(os.path.join("output/error_log", ef), "r") as f:
                    content = f.read()
                    if str(p)[:-1] in content:
                        found = True
                        break
            filter_bool3.append(not found)

    # Combine the three filters
    filter_bool = [b1 and b2 and b3 for b1, b2, b3 in zip(filter_bool1, filter_bool2, filter_bool3)]

    # Print the number of parameters kept and discarded for each filter
    kept_count = sum(filter_bool)
    discarded_count = len(params_all) - kept_count
    print("\n***Parameters filtering***")
    print('--------------------------')
    print(f"Parameters kept after filtering: {kept_count}/{len(params_all)}")
    print(f"Parameters discarded by filtering: {discarded_count}/{len(params_all)}")
    # Print the number of parameters kept and discarded for each filter
    print(f"Parameters discarded by manual filtering: {(len(params_all) - sum(filter_bool1))}/{len(params_all)}")
    print(f"Parameters discarded by summary filtering: {(len(params_all) - sum(filter_bool2))}/{len(params_all)}")
    print(f"Parameters discarded by error log filtering: {(len(params_all) - sum(filter_bool3))}/{len(params_all)}")

    return filter_bool

def filter_params_wrapper(params_all, summary_filepath=None):
    # Filter the parameters if a summary file is provided
    if isinstance(summary_filepath, str):
        print("\n")
        filter_bool = filter_params(params_all, summary_filepath)
        filtered_params = [p for p, keep in zip(params_all, filter_bool) if keep]
        discarded_params = [p for p, keep in zip(params_all, filter_bool) if not keep]
        print('\n\n***List of discarded parameter sets***')
        print('--------------------------------------')
        print_parameter_sets(discarded_params)
    else:
        filtered_params = params_all
    return filtered_params


def parallel_dfl_training(task_id, n_jobs):
    """
    This function runs the DFLopti optimization in parallel.
    Args:
        task_id: the ID of the task to run
        n_tasks: the total number of tasks to run in parallel
        summary_filepath: path to the summary file to filter the parameters that were already run (optional)
    """
    # Read the parameters from the pickle file
    with open(f"output/task_parameters/task_{task_id}_params.pkl", "rb") as f:
        params = pickle.load(f)

    # for better visualization, to check the parameters
    params_df = pd.DataFrame(params)
    hpp_df = pd.DataFrame(params_df['hpp'].tolist())
    print('\n\n***List of parameter sets to be run for this task***')
    print('------------------------------------------------\n')
    print(f'Number of parameter sets for this task: {len(params)}')
    print(pd.concat([params_df["thermal_model"], hpp_df.reindex(columns=['nb_layers', 'nb_neurons']), params_df[["loss_metric", "ems_relax", "seed", "stds", "test_only"]]], axis=1).to_string())
    # Process the specific parameter for this task
    Parallel(n_jobs=n_jobs, verbose=10, backend="loky")(delayed(safe_process_task)(param) for param in params)

    # For debugging, parallelism is to be avoided.
    # for param in params:
    #     print(f"\n\n***Processing parameters: {param}***")
    #     process_task(param)


def paper_parameter_sets1(n_tasks):
    """
    This function selects the run configurations to reproduce the results of the paper.
    """
    params = [
        # test only RC model
        {'ems_relax': np.nan, 'gamma': np.nan,
         'hpp': {'target': 'Zone Mean Air Temperature(t+1)'}, 'learning_rate': np.nan, 'loss_metric': 'expost+',
          'nb_epochs_max': 100, 'seed': 0, 'snr': np.nan,
         'stds': np.nan, 'std_requires_grad': False, 's': np.nan, 'test_only': True, 'thermal_model': 'rcmodel', 'update_frequency': np.nan,
         'warm_start': "True"},
        # test only NN1 model
        {'ems_relax': np.nan, 'gamma': np.nan,
         'hpp': {'activation': 'ReLU()', 'nb_inputs': 11, 'nb_layers': 1, 'nb_neurons': 2, 'sparse': False,
                 'target': 'Zone Mean Air Temperature(t+1)'}, 'learning_rate': np.nan, 'loss_metric': 'expost+',
          'nb_epochs_max': 100, 'seed': 0, 'snr': np.nan,
         'stds': np.nan, 'std_requires_grad': False, 's': np.nan, 'test_only': True, 'thermal_model': 'nn', 'update_frequency': np.nan,
         'warm_start': "True"},
        # test only NN2 model
        {'ems_relax': np.nan, 'gamma': np.nan,
         'hpp': {'activation': 'ReLU()', 'nb_inputs': 11, 'nb_layers': 1, 'nb_neurons': 5, 'sparse': False,
                 'target': 'Zone Mean Air Temperature(t+1)'}, 'learning_rate': np.nan, 'loss_metric': 'expost+',
          'nb_epochs_max': 100, 'seed': 0, 'snr': np.nan,
         'stds': np.nan, 'std_requires_grad': False, 's': np.nan, 'test_only': True, 'thermal_model': 'nn', 'update_frequency': np.nan,
         'warm_start': "True"},
        # test only NN3 model
        {'ems_relax': np.nan, 'gamma': np.nan,
         'hpp': {'activation': 'ReLU()', 'nb_inputs': 11, 'nb_layers': 1, 'nb_neurons': 10, 'sparse': False,
                 'target': 'Zone Mean Air Temperature(t+1)'}, 'learning_rate': np.nan, 'loss_metric': 'expost+',
          'nb_epochs_max': 100, 'seed': 0, 'snr': np.nan,
         'stds': np.nan, 'std_requires_grad': False, 's': np.nan, 'test_only': True, 'thermal_model': 'nn', 'update_frequency': np.nan,
         'warm_start': "True"},
        # # RC model with QP relaxation
        # {'ems_relax': 'qp', 'gamma': 0.98,
        #  'hpp': {'target': 'Zone Mean Air Temperature(t+1)'}, 'learning_rate': 1e-2, 'loss_metric': 'hierarchical_weighted_mae',
        #   'nb_epochs_max': 100, 'seed': 1, 'snr': 0,
        #  'stds': 0, 'std_requires_grad': False, 's': 1, 'test_only': False, 'thermal_model': 'rcmodel', 'update_frequency': 1,
        #  'warm_start': "True"},
        # # NN1 model with QP relaxation
        # {'ems_relax': 'qp', 'gamma': 0.98,
        #  'hpp': {'activation': 'ReLU()', 'nb_inputs': 11, 'nb_layers': 1, 'nb_neurons': 2, 'sparse': False,
        #          'target': 'Zone Mean Air Temperature(t+1)'}, 'learning_rate': 1e-3, 'loss_metric': 'hierarchical_weighted_mae',
        #   'nb_epochs_max': 100, 'seed': 4, 'snr': 0,
        #  'stds': 0, 'std_requires_grad': False, 's': 1, 'test_only': False, 'thermal_model': 'nn', 'update_frequency': 1, 'warm_start': "True"},
        # # NN1 model with FB relaxation
        # {'ems_relax': 'fixed_bin', 'gamma': 0.98,
        #  'hpp': {'activation': 'ReLU()', 'nb_inputs': 11, 'nb_layers': 1, 'nb_neurons': 2, 'sparse': False,
        #          'target': 'Zone Mean Air Temperature(t+1)'}, 'learning_rate': 1e-3, 'loss_metric': 'hierarchical_weighted_mae',
        #   'nb_epochs_max': 100, 'seed': 4, 'snr': 0,
        #  'stds': 0, 'std_requires_grad': False, 's': 1, 'test_only': False, 'thermal_model': 'nn', 'update_frequency': 1, 'warm_start': "True"},
        # # NN1 model with SS relaxation  (s = 1)
        # {'ems_relax': 'ss', 'gamma': 0.98,
        #  'hpp': {'activation': 'ReLU()', 'nb_inputs': 11, 'nb_layers': 1, 'nb_neurons': 2, 'sparse': False,
        #          'target': 'Zone Mean Air Temperature(t+1)'}, 'learning_rate': 1e-3, 'loss_metric': 'expost+',
        #   'nb_epochs_max': 100, 'seed': 2, 'snr': 0,
        #  'stds': 0.05, 'std_requires_grad': False, 's': 1, 'test_only': False, 'thermal_model': 'nn', 'update_frequency': 1,
        #  'warm_start': "True"},
        # # NN2 model with QP relaxation
        # {'ems_relax': 'qp', 'gamma': 0.98,
        #  'hpp': {'activation': 'ReLU()', 'nb_inputs': 11, 'nb_layers': 1, 'nb_neurons': 5, 'sparse': False,
        #          'target': 'Zone Mean Air Temperature(t+1)'}, 'learning_rate': 1e-3, 'loss_metric': 'hierarchical_weighted_mae',
        #   'nb_epochs_max': 100, 'seed': 1, 'snr': 0,
        #  'stds': 0, 'std_requires_grad': False, 's': 1, 'test_only': False, 'thermal_model': 'nn', 'update_frequency': 1, 'warm_start': "True"},
        # # NN2 model with FB relaxation
        # {'ems_relax': 'fixed_bin', 'gamma': 0.98,
        #  'hpp': {'activation': 'ReLU()', 'nb_inputs': 11, 'nb_layers': 1, 'nb_neurons': 5, 'sparse': False,
        #          'target': 'Zone Mean Air Temperature(t+1)'}, 'learning_rate': 1e-3, 'loss_metric': 'hierarchical_weighted_mae',
        #   'nb_epochs_max': 100, 'seed': 0, 'snr': 0,
        #  'stds': 0, 'std_requires_grad': False, 's': 1, 'test_only': False, 'thermal_model': 'nn', 'update_frequency': 1, 'warm_start': "True"},
        # # NN2 model with SS relaxation  (s = 1)
        # {'ems_relax': 'ss', 'gamma': 0.98,
        #  'hpp': {'activation': 'ReLU()', 'nb_inputs': 11, 'nb_layers': 1, 'nb_neurons': 5, 'sparse': False,
        #          'target': 'Zone Mean Air Temperature(t+1)'}, 'learning_rate': 1e-3, 'loss_metric': 'expost+',
        #   'nb_epochs_max': 100, 'seed': 2, 'snr': 0,
        #  'stds': 0.01, 'std_requires_grad': False, 's': 1, 'test_only': False, 'thermal_model': 'nn', 'update_frequency': 1, 'warm_start': "True"},
        # # NN3 model with QP relaxation
        # {'ems_relax': 'qp', 'gamma': 0.98,
        #  'hpp': {'activation': 'ReLU()', 'nb_inputs': 11, 'nb_layers': 1, 'nb_neurons': 10, 'sparse': False,
        #          'target': 'Zone Mean Air Temperature(t+1)'}, 'learning_rate': 1e-3, 'loss_metric': 'hierarchical_weighted_mae',
        #   'nb_epochs_max': 100, 'seed': 0, 'snr': 0,
        #  'stds': 0, 'std_requires_grad': False, 's': 1, 'test_only': False, 'thermal_model': 'nn', 'update_frequency': 1, 'warm_start': "True"},

        ### TEST OF SS WITH DIFFERENT S VALUES
        # # NN1 model with SS relaxation (s = 2)
        # {'ems_relax': 'ss', 'gamma': 0.98,
        #  'hpp': {'activation': 'ReLU()', 'nb_inputs': 11, 'nb_layers': 1, 'nb_neurons': 2, 'sparse': False,
        #          'target': 'Zone Mean Air Temperature(t+1)'}, 'learning_rate': 1e-3, 'loss_metric': 'expost+',
        #   'nb_epochs_max': 100, 'seed': 2, 'snr': 0,
        #  'stds': 0.05, 'std_requires_grad': False, 's': 2, 'test_only': False, 'thermal_model': 'nn', 'update_frequency': 1,
        #  'warm_start': "True"},
        # # NN2 model with SS relaxation (s = 2)
        # {'ems_relax': 'ss', 'gamma': 0.98,
        #  'hpp': {'activation': 'ReLU()', 'nb_inputs': 11, 'nb_layers': 1, 'nb_neurons': 5, 'sparse': False,
        #          'target': 'Zone Mean Air Temperature(t+1)'}, 'learning_rate': 1e-3, 'loss_metric': 'expost+',
        #   'nb_epochs_max': 100, 'seed': 2, 'snr': 0,
        #  'stds': 0.01, 'std_requires_grad': False, 's': 2, 'test_only': False, 'thermal_model': 'nn', 'update_frequency': 1, 'warm_start': "True"},
        # # NN1 model with SS relaxation (s = 5)
        # {'ems_relax': 'ss', 'gamma': 0.98,
        #  'hpp': {'activation': 'ReLU()', 'nb_inputs': 11, 'nb_layers': 1, 'nb_neurons': 2, 'sparse': False,
        #          'target': 'Zone Mean Air Temperature(t+1)'}, 'learning_rate': 1e-3, 'loss_metric': 'expost+',
        #   'nb_epochs_max': 100, 'seed': 2, 'snr': 0,
        #  'stds': 0.05, 'std_requires_grad': False, 's': 5, 'test_only': False, 'thermal_model': 'nn', 'update_frequency': 1,
        #  'warm_start': "True"},
        # # NN2 model with SS relaxation (s = 5)
        # {'ems_relax': 'ss', 'gamma': 0.98,
        #  'hpp': {'activation': 'ReLU()', 'nb_inputs': 11, 'nb_layers': 1, 'nb_neurons': 5, 'sparse': False,
        #          'target': 'Zone Mean Air Temperature(t+1)'}, 'learning_rate': 1e-3, 'loss_metric': 'expost+',
        #   'nb_epochs_max': 100, 'seed': 2, 'snr': 0,
        #  'stds': 0.01, 'std_requires_grad': False, 's': 5, 'test_only': False, 'thermal_model': 'nn', 'update_frequency': 1, 'warm_start': "True"},
        # # NN1 model with SS relaxation (s = 10)
        # {'ems_relax': 'ss', 'gamma': 0.98,
        #  'hpp': {'activation': 'ReLU()', 'nb_inputs': 11, 'nb_layers': 1, 'nb_neurons': 2, 'sparse': False,
        #          'target': 'Zone Mean Air Temperature(t+1)'}, 'learning_rate': 1e-3, 'loss_metric': 'expost+',
        #   'nb_epochs_max': 100, 'seed': 2, 'snr': 0,
        #  'stds': 0.05, 'std_requires_grad': False, 's': 10, 'test_only': False, 'thermal_model': 'nn', 'update_frequency': 1,
        #  'warm_start': "True"},
        # # NN2 model with SS relaxation (s = 10)
        # {'ems_relax': 'ss', 'gamma': 0.98,
        #  'hpp': {'activation': 'ReLU()', 'nb_inputs': 11, 'nb_layers': 1, 'nb_neurons': 5, 'sparse': False,
        #          'target': 'Zone Mean Air Temperature(t+1)'}, 'learning_rate': 1e-3, 'loss_metric': 'expost+',
        #   'nb_epochs_max': 100, 'seed': 2, 'snr': 0,
        #  'stds': 0.01, 'std_requires_grad': False, 's': 10, 'test_only': False, 'thermal_model': 'nn',
        #  'update_frequency': 1, 'warm_start': "True"},

        # TRAINABLE STD

    ]
    # get the initial folder path
    for p in params:
        p["initial_model_folderpath"] = get_initial_model_folderpath(p)
    # Split the parameters into chunks for each task and save them
    split_params(params, n_tasks)


def paper_parameter_sets2(n_tasks):
    ems_relaxations = ["fixed_bin", "qp", "ss"]  # "fixed_bin", "qp", "ss"
    nb_epochs_max = 100
    seeds = range(0, 5)  # seed for reproducibility

    hyperparameters = [
        ("nn", {"target": "Zone Mean Air Temperature(t+1)", "nb_layers": 1, "nb_neurons": 2, "nb_inputs": 11,
                          "activation": "ReLU()", "sparse": False}),
        ("nn", {"target": "Zone Mean Air Temperature(t+1)", "nb_layers": 1, "nb_neurons": 5, "nb_inputs": 11,
                           "activation": "ReLU()", "sparse": False}),
        ("nn", {"target": "Zone Mean Air Temperature(t+1)", "nb_layers": 1, "nb_neurons": 10, "nb_inputs": 11,
                "activation": "ReLU()", "sparse": False}),
        ("rcmodel", {"target": "Zone Mean Air Temperature(t+1)"}),
    ]

    # Loss metric is given by the ems_relaxation: "qp" and "fixed_bin" -> "hierarchical_weighted_mae", "ss" -> "expost+"
    # loss_metrics = ["expost+", "hierarchical_weighted_mae"]
    learning_rates = [1e-3] # [1e-3, 5e-3, 1e-2]
    gamma = 0.98
    update_frequency = 1  # update the weights every x medoids
    # Whether to warm start the NN/RC training with the initial model (trained on historical data) or not
    # If warm_start is False and test_only is False, then the DFL model is trained from scratch
    # "Noise" means warm start with noise of amplitude signal-to-noise ratio = SNR on the weights
    warm_start = "Noise"  # "Noise" or "True" or "False"
    snr = 1000
    # value of the std and whether the std is a learnable parameter (if std init = cst) or not
    stds = [
        (0, False),
        (0.01, False),
        (0.05, False), (0.1, False),
            (0.01, True), (0.05, True), (0.1, True),
            (lambda x: x/2, False), (lambda x: x/10, False),
            (lambda x: x/2, True), (lambda x: x/10, True)
            ]
    S = [1]  # number of samples to approximate the gradient with stochastic smoothing
    test_only_l = [False]  # True, False

    params = []
    for test_only in test_only_l:
        for thermal_model, hpp in hyperparameters:
            # If test_only is True, we do not need to specify ems_relax, learning_rate, stds, and seed
            if test_only:
                params.append({
                    "thermal_model": thermal_model,
                    "hpp": hpp,
                    "loss_metric": "expost+",
                    "ems_relax": np.nan,  # no need to specify ems_relax for test_only
                    "learning_rate": np.nan,  # no need to specify ems_relax for test_only
                    "gamma": np.nan,
                    "update_frequency": np.nan,
                    "warm_start": warm_start,
                    "seed": 0,
                    "snr": np.nan,
                    "stds": np.nan,
                    "s": np.nan,
                    "test_only": test_only,
                    "nb_epochs_max": 1,
                })
                continue

            for seed in seeds:
                for ems_relax, learning_rate in product(ems_relaxations, learning_rates):
                    loss_metric = "expost+" if ems_relax == "ss" else "hierarchical_weighted_mae"

                    for (std, rg), s in product(stds, S):
                        params.append({
                            "thermal_model": thermal_model,
                            "hpp": hpp,
                            "loss_metric": loss_metric,
                            "ems_relax": ems_relax,
                            "learning_rate": learning_rate,
                            "gamma": gamma,
                            "update_frequency": update_frequency,
                            "warm_start": warm_start,
                            "seed": seed,
                            "snr": snr,
                            "stds": std,
                            "std_requires_grad": rg,
                            "s": s,
                            "test_only": test_only,
                            "nb_epochs_max": nb_epochs_max,
                        })

    # get the initial folder path
    for p in params:
        p["initial_model_folderpath"] = get_initial_model_folderpath(p)
    # Modify the parameters manually if needed to improve the performance of the training
    params_all = manual_modification_of_parameters(params)
    # Filter the parameters based on the parameters already run (successfully in summary.xlsx or failed in error_log)
    filtered_params = filter_params_wrapper(params_all, "data/SmallOffice/Models/cvxpylayer/summary.xlsx")
    # Split the parameters into chunks for parallel processing by the array job and save them to a file
    split_params(filtered_params, n_tasks)


def paper_parameter_sets3(n_tasks):
    ems_relaxations = ["ss"]  # "fixed_bin", "qp", "ss"
    nb_epochs_max = 100
    seeds = range(0, 5)  # seed for reproducibility

    hyperparameters = [
        ("nn", {"target": "Zone Mean Air Temperature(t+1)", "nb_layers": 1, "nb_neurons": 2, "nb_inputs": 11,
                          "activation": "ReLU()", "sparse": False}),
        ("nn", {"target": "Zone Mean Air Temperature(t+1)", "nb_layers": 1, "nb_neurons": 5, "nb_inputs": 11,
                           "activation": "ReLU()", "sparse": False}),
        # ("nn", {"target": "Zone Mean Air Temperature(t+1)", "nb_layers": 1, "nb_neurons": 10, "nb_inputs": 11,
        #         "activation": "ReLU()", "sparse": False}),
        # ("rcmodel", {"target": "Zone Mean Air Temperature(t+1)"}),
    ]

    # Loss metric is given by the ems_relaxation: "qp" and "fixed_bin" -> "hierarchical_weighted_mae", "ss" -> "expost+"
    # loss_metrics = ["expost+", "hierarchical_weighted_mae"]
    learning_rates = [1e-3] # [1e-3, 5e-3, 1e-2]
    gamma = 0.98
    update_frequency = 1  # update the weights every x medoids
    # Whether to warm start the NN/RC training with the initial model (trained on historical data) or not
    # If warm_start is False and test_only is False, then the DFL model is trained from scratch
    # "Noise" means warm start with noise of amplitude signal-to-noise ratio = SNR on the weights
    warm_start = "Noise"  # "Noise" or "True" or "False"
    snr = 1000
    # value of the std and whether the std is a learnable parameter (if std init = cst) or not
    stds = [
        # (0, False),
        (0.01, False),
        # (0.05, False), (0.1, False),
        #     (0.01, True), (0.05, True), (0.1, True),
        #     (lambda x: x/2, False), (lambda x: x/10, False),
        #     (lambda x: x/2, True), (lambda x: x/10, True)
            ]
    S = [2, 5, 10]  # number of samples to approximate the gradient with stochastic smoothing
    test_only_l = [False]  # True, False

    params = []
    for test_only in test_only_l:
        for thermal_model, hpp in hyperparameters:
            # If test_only is True, we do not need to specify ems_relax, learning_rate, stds, and seed
            if test_only:
                params.append({
                    "thermal_model": thermal_model,
                    "hpp": hpp,
                    "loss_metric": "expost+",
                    "ems_relax": np.nan,  # no need to specify ems_relax for test_only
                    "learning_rate": np.nan,  # no need to specify ems_relax for test_only
                    "gamma": np.nan,
                    "update_frequency": np.nan,
                    "warm_start": warm_start,
                    "seed": 0,
                    "snr": np.nan,
                    "stds": np.nan,
                    "s": np.nan,
                    "test_only": test_only,
                    "nb_epochs_max": 1,
                })
                continue

            for seed in seeds:
                for ems_relax, learning_rate in product(ems_relaxations, learning_rates):
                    loss_metric = "expost+" if ems_relax == "ss" else "hierarchical_weighted_mae"

                    for (std, rg), s in product(stds, S):
                        params.append({
                            "thermal_model": thermal_model,
                            "hpp": hpp,
                            "loss_metric": loss_metric,
                            "ems_relax": ems_relax,
                            "learning_rate": learning_rate,
                            "gamma": gamma,
                            "update_frequency": update_frequency,
                            "warm_start": warm_start,
                            "seed": seed,
                            "snr": snr,
                            "stds": std,
                            "std_requires_grad": rg,
                            "s": s,
                            "test_only": test_only,
                            "nb_epochs_max": nb_epochs_max,
                        })

    # get the initial folder path
    for p in params:
        p["initial_model_folderpath"] = get_initial_model_folderpath(p)
    # Modify the parameters manually if needed to improve the performance of the training
    params_all = manual_modification_of_parameters(params)
    # Filter the parameters based on the parameters already run (successfully in summary.xlsx or failed in error_log)
    filtered_params = filter_params_wrapper(params_all, "data/SmallOffice/Models/cvxpylayer/summary.xlsx")
    # Split the parameters into chunks for parallel processing by the array job and save them to a file
    split_params(filtered_params, n_tasks)


def paper_parameter_sets4(n_tasks):
    """
    This function selects the run configurations to reproduce the results of the paper.
    The runs that requires other runs as warm-start are kept here.
    """
    params = [
        # NN1 model with QP relaxation (tight)
        {'ems_relax': 'qp', 'gamma': 0.98,
         'hpp': {'activation': 'ReLU()', 'nb_inputs': 11, 'nb_layers': 1, 'nb_neurons': 2, 'sparse': False,
                 'target': 'Zone Mean Air Temperature(t+1)'}, 'learning_rate': 1e-3,
         'loss_metric': 'hierarchical_weighted_mae',
          'nb_epochs_max': 100, 'seed': 4, 'snr': 1000,
         'stds': 0, 'std_requires_grad': False, 's': 1, 'test_only': True, 'thermal_model': 'nn',
         'update_frequency': 1, 'warm_start': "dfl_tight"},
        # NN1 model with FB relaxation (tight)
        {'ems_relax': 'fixed_bin', 'gamma': 0.98,
         'hpp': {'activation': 'ReLU()', 'nb_inputs': 11, 'nb_layers': 1, 'nb_neurons': 2, 'sparse': False,
                 'target': 'Zone Mean Air Temperature(t+1)'}, 'learning_rate': 1e-3,
         'loss_metric': 'hierarchical_weighted_mae',
          'nb_epochs_max': 100, 'seed': 1, 'snr': 1000,
         'stds': 0, 'std_requires_grad': False, 's': 1, 'test_only': True, 'thermal_model': 'nn',
         'update_frequency': 1, 'warm_start': "dfl_tight"},
        # NN1 model with SS relaxation (tight)
        {'ems_relax': 'ss', 'gamma': 0.98,
         'hpp': {'activation': 'ReLU()', 'nb_inputs': 11, 'nb_layers': 1, 'nb_neurons': 2, 'sparse': False,
                 'target': 'Zone Mean Air Temperature(t+1)'}, 'learning_rate': 1e-3, 'loss_metric': 'expost+',
          'nb_epochs_max': 100, 'seed': 1, 'snr': 1000,
         'stds': 0.01, 'std_requires_grad': False, 's': 1, 'test_only': True, 'thermal_model': 'nn',
         'update_frequency': 1, 'warm_start': "dfl_tight"},
        # NN2 model with QP relaxation (tight)
        {'ems_relax': 'qp', 'gamma': 0.98,
         'hpp': {'activation': 'ReLU()', 'nb_inputs': 11, 'nb_layers': 1, 'nb_neurons': 5, 'sparse': False,
                 'target': 'Zone Mean Air Temperature(t+1)'}, 'learning_rate': 1e-3,
         'loss_metric': 'hierarchical_weighted_mae',
          'nb_epochs_max': 100, 'seed': 0, 'snr': 1000,
         'stds': 0, 'std_requires_grad': False, 's': 1, 'test_only': True, 'thermal_model': 'nn',
         'update_frequency': 1, 'warm_start': "dfl_tight"},
        # NN2 model with FB relaxation (tight)
        {'ems_relax': 'fixed_bin', 'gamma': 0.98,
         'hpp': {'activation': 'ReLU()', 'nb_inputs': 11, 'nb_layers': 1, 'nb_neurons': 5, 'sparse': False,
                 'target': 'Zone Mean Air Temperature(t+1)'}, 'learning_rate': 1e-3,
         'loss_metric': 'hierarchical_weighted_mae',
          'nb_epochs_max': 100, 'seed': 4, 'snr': 1000,
         'stds': 0, 'std_requires_grad': False, 's': 1, 'test_only': True, 'thermal_model': 'nn',
         'update_frequency': 1, 'warm_start': "dfl_tight"},
        # NN2 model with SS relaxation (tight)
        {'ems_relax': 'ss', 'gamma': 0.98,
         'hpp': {'activation': 'ReLU()', 'nb_inputs': 11, 'nb_layers': 1, 'nb_neurons': 5, 'sparse': False,
                 'target': 'Zone Mean Air Temperature(t+1)'}, 'learning_rate': 1e-3, 'loss_metric': 'expost+',
          'nb_epochs_max': 100, 'seed': 3, 'snr': 1000,
         'stds': 0.01, 'std_requires_grad': False, 's': 1, 'test_only': True, 'thermal_model': 'nn',
         'update_frequency': 1, 'warm_start': "dfl_tight"},
        # NN3 model with QP relaxation (tight)
        {'ems_relax': 'qp', 'gamma': 0.98,
         'hpp': {'activation': 'ReLU()', 'nb_inputs': 11, 'nb_layers': 1, 'nb_neurons': 10, 'sparse': False,
                 'target': 'Zone Mean Air Temperature(t+1)'}, 'learning_rate': 1e-3,
         'loss_metric': 'hierarchical_weighted_mae',
          'nb_epochs_max': 100, 'seed': 3, 'snr': 1000,
         'stds': 0, 'std_requires_grad': False, 's': 1, 'test_only': True, 'thermal_model': 'nn',
         'update_frequency': 1, 'warm_start': "dfl_tight"},
        # NN1 model with QP relaxation (not tight)
        {'ems_relax': 'qp', 'gamma': 0.98,
         'hpp': {'activation': 'ReLU()', 'nb_inputs': 11, 'nb_layers': 1, 'nb_neurons': 2, 'sparse': False,
                 'target': 'Zone Mean Air Temperature(t+1)'}, 'learning_rate': 1e-3,
         'loss_metric': 'hierarchical_weighted_mae',
          'nb_epochs_max': 100, 'seed': 4, 'snr': 1000,
         'stds': 0, 'std_requires_grad': False, 's': 1, 'test_only': True, 'thermal_model': 'nn',
         'update_frequency': 1, 'warm_start': "dfl_not_tight"},
        # NN1 model with FB relaxation (not tight)
        {'ems_relax': 'fixed_bin', 'gamma': 0.98,
         'hpp': {'activation': 'ReLU()', 'nb_inputs': 11, 'nb_layers': 1, 'nb_neurons': 2, 'sparse': False,
                 'target': 'Zone Mean Air Temperature(t+1)'}, 'learning_rate': 1e-3,
         'loss_metric': 'hierarchical_weighted_mae',
          'nb_epochs_max': 100, 'seed': 1, 'snr': 1000,
         'stds': 0, 'std_requires_grad': False, 's': 1, 'test_only': True, 'thermal_model': 'nn',
         'update_frequency': 1, 'warm_start': "dfl_not_tight"},
        # NN1 model with SS relaxation (not tight)
        {'ems_relax': 'ss', 'gamma': 0.98,
         'hpp': {'activation': 'ReLU()', 'nb_inputs': 11, 'nb_layers': 1, 'nb_neurons': 2, 'sparse': False,
                 'target': 'Zone Mean Air Temperature(t+1)'}, 'learning_rate': 1e-3, 'loss_metric': 'expost+',
          'nb_epochs_max': 100, 'seed': 1, 'snr': 1000,
         'stds': 0.05, 'std_requires_grad': False, 's': 1,
         'test_only': True, 'thermal_model': 'nn', 'update_frequency': 1, 'warm_start': "dfl_not_tight"},
        # NN2 model with QP relaxation (not tight)
        {'ems_relax': 'qp', 'gamma': 0.98,
         'hpp': {'activation': 'ReLU()', 'nb_inputs': 11, 'nb_layers': 1, 'nb_neurons': 5, 'sparse': False,
                 'target': 'Zone Mean Air Temperature(t+1)'}, 'learning_rate': 1e-3,
         'loss_metric': 'hierarchical_weighted_mae',
          'nb_epochs_max': 100, 'seed': 0, 'snr': 1000,
         'stds': 0, 'std_requires_grad': False, 's': 1, 'test_only': True, 'thermal_model': 'nn',
         'update_frequency': 1, 'warm_start': "dfl_not_tight"},
        # NN2 model with FB relaxation (not tight)
        {'ems_relax': 'fixed_bin', 'gamma': 0.98,
         'hpp': {'activation': 'ReLU()', 'nb_inputs': 11, 'nb_layers': 1, 'nb_neurons': 5, 'sparse': False,
                 'target': 'Zone Mean Air Temperature(t+1)'}, 'learning_rate': 1e-3,
         'loss_metric': 'hierarchical_weighted_mae',
          'nb_epochs_max': 100, 'seed': 4, 'snr': 1000,
         'stds': 0, 'std_requires_grad': False, 's': 1, 'test_only': True, 'thermal_model': 'nn',
         'update_frequency': 1, 'warm_start': "dfl_not_tight"},
        # NN2 model with SS relaxation (not tight)
        {'ems_relax': 'ss', 'gamma': 0.98,
         'hpp': {'activation': 'ReLU()', 'nb_inputs': 11, 'nb_layers': 1, 'nb_neurons': 5, 'sparse': False,
                 'target': 'Zone Mean Air Temperature(t+1)'}, 'learning_rate': 1e-3, 'loss_metric': 'expost+',
          'nb_epochs_max': 100, 'seed': 3, 'snr': 1000,
         'stds': 0.01, 'std_requires_grad': False, 's': 1, 'test_only': True, 'thermal_model': 'nn',
         'update_frequency': 1, 'warm_start': "dfl_not_tight"},
        # NN3 model with QP relaxation (not tight)
        {'ems_relax': 'qp', 'gamma': 0.98,
         'hpp': {'activation': 'ReLU()', 'nb_inputs': 11, 'nb_layers': 1, 'nb_neurons': 10, 'sparse': False,
                 'target': 'Zone Mean Air Temperature(t+1)'}, 'learning_rate': 1e-3,
         'loss_metric': 'hierarchical_weighted_mae',
          'nb_epochs_max': 100, 'seed': 3, 'snr': 1000,
         'stds': 0, 'std_requires_grad': False, 's': 1, 'test_only': True, 'thermal_model': 'nn',
         'update_frequency': 1, 'warm_start': "dfl_not_tight"},
        ]

    # get the initial folder path
    for p in params:
        p["initial_model_folderpath"] = get_initial_model_folderpath(p)
    # Split the parameters into chunks for each task and save them
    split_params(params, n_tasks)



