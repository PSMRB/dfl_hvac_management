"""
Create the list of parameters to be executed in parallel for the DFL optimization.
"""

from itertools import product, repeat
from joblib import Parallel, delayed
import math
import numpy as np
import os
import pandas as pd
import pickle
import traceback
import warnings
import src.library.Common as c
from src.library.DeclarativeNN import DeclarativeNN

####################################
### Mofify warnings format
####################################

def warning_on_one_line(message, category, filename, lineno, file=None, line=None):
    # Format as filename:lineno: category: message
    return f'\n{filename}:{lineno}: {category.__name__}:\n{message}\n'

warnings.formatwarning = warning_on_one_line

####################################
### Parameters
####################################
# N.B., the file 18zones_ASHRAE901_OfficeMedium_STD2019_Denver.idf does not account for daylight saving time.
# Also, it is recommended to perform all the operation in UTC time and convert the results to the local time zone
# only for visualization purposes.

def get_medoids_and_labels():
    # Read the medoids days and reorder them to create a smooth transition between the days
    folderpath = "data/optimization_parameters/2006-2020"
    medoid_labels = pd.read_csv(os.path.join(folderpath, "Labels10Days.csv"), index_col=0,
                                        parse_dates=True)
    medoids = pd.read_csv(os.path.join(folderpath, "Clusters10Days.csv"), index_col=0, parse_dates=True)
    medoids = medoids.sort_values(by="Mean Tamb", ascending=True)
    m_idx = medoids.index
    nb_medoids = len(m_idx)
    cycle = [m_idx[i] for i in range(nb_medoids) if i % 2 == 0] + [m_idx[i] for i in range(nb_medoids, -1, -1) if i % 2 == 1]
    medoids = medoids.loc[cycle]
    print("\n\n***List of medoids***")
    print("---------------------")
    print(medoids.to_string())

    return medoids, medoid_labels

def create_parameter_list():

    medoids, medoid_labels = get_medoids_and_labels()

    ems_relaxations = ["ss"]  # "fixed_bin", "qp", "ss"
    nb_epochs_max = 100
    snr = 0
    seeds = range(0, 5)  # seed for reproducibility

    hyperparameters = [
        # ("nn", {"target": "Zone Mean Air Temperature(t+1)", "nb_layers": 1, "nb_neurons": 2, "nb_inputs": 11,
        #                   "activation": "ReLU()", "sparse": False}),
        # ("nn", {"target": "Zone Mean Air Temperature(t+1)", "nb_layers": 1, "nb_neurons": 5, "nb_inputs": 11,
        #                    "activation": "ReLU()", "sparse": False}),
        # ("nn", {"target": "Zone Mean Air Temperature(t+1)", "nb_layers": 1, "nb_neurons": 10, "nb_inputs": 11,
        #         "activation": "ReLU()", "sparse": False}),
        ("rcmodel", {"target": "Zone Mean Air Temperature(t+1)"}),
    ]

    # Loss metric is given by the ems_relaxation: "qp" and "fixed_bin" -> "hierarchical_weighted_mae", "ss" -> "expost+"
    # loss_metrics = ["expost+", "hierarchical_weighted_mae"]
    learning_rates = [2e-3, 2e-2] # [1e-3, 5e-3, 1e-2]
    gamma = 0.98
    update_frequency = 1  # update the weights every x samples
    # Whether to warm start the NN training as the RC model or not
    warm_start = "True"  # "Noise" or "True" or "False"
    stds = [0.25, 0.5, 1]  # 0.01, 0.02, 0.03
    test_only_l = [False]  # True, False

    params = []
    for test_only in test_only_l:
        for thermal_model, hpp in hyperparameters:
            for seed in seeds:
                # If test_only is True, we do not need to specify ems_relax, learning_rate, stds
                if test_only:
                    params.append({
                        "thermal_model": thermal_model,
                        "hpp": hpp,
                        "loss_metric": "expost+",
                        "ems_relax": np.nan, # no need to specify ems_relax for test_only
                        "learning_rate": np.nan, # no need to specify ems_relax for test_only
                        "gamma": np.nan,
                        "update_frequency": np.nan,
                        "warm_start": np.nan,
                        "seed": seed,
                        "snr": np.nan,
                        "stds": np.nan,
                        "test_only": test_only,
                        "medoids": medoids,
                        "medoid_labels": medoid_labels,
                        "nb_epochs_max": nb_epochs_max,
                    })
                    continue
                for ems_relax, learning_rate in product(ems_relaxations, learning_rates):
                    if "rcmodel" in thermal_model and ems_relax == "fixed_bin":
                        warnings.warn("RC model must be used with QP or SS relaxation. This run configuration was skipped.")
                        continue

                    loss_metric = "expost+" if ems_relax == "ss" else "hierarchical_weighted_mae"

                    for std in stds:
                        if ems_relax != "ss":
                            std = 0

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
                            "test_only": test_only,
                            "medoids": medoids,
                            "medoid_labels": medoid_labels,
                            "nb_epochs_max": nb_epochs_max,
                        })

                        # if ems_relax is not ss, no need to iterate over stds
                        if ems_relax != "ss":
                            break

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

    # Compute start and end indices for each task and save the parameters
    for task_id in range(n_tasks):
        start = task_id * base + min(task_id, rem)
        end = start + base + (1 if task_id < rem else 0)
        task_params = params_all[start:end]
        # print task parameters for debugging
        print(f"\n\n***Parameters for task {task_id}***")
        print('---------------------------------')
        print_parameter_sets(task_params)
        # Save the parameters for this task
        task_filepath = os.path.join("output/task_parameters", f"task_{task_id}_params.pkl")
        if not os.path.exists(os.path.dirname(task_filepath)):
            os.makedirs(os.path.dirname(task_filepath))
        with open(task_filepath, "wb") as f:
            pickle.dump(task_params, f)

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
    hpp = params["hpp"]
    loss_metric = params["loss_metric"]
    seed = params["seed"]
    test_only = params["test_only"]
    medoids = params["medoids"]
    medoid_labels = params["medoid_labels"]
    nb_epochs_max = params["nb_epochs_max"]

    seed, _, _ = c.make_reproducible(seed)
    ### to test ITO models
    dnn = DeclarativeNN()
    if test_only:
        dnn.test_ito(loss_metric, medoids, medoid_labels, hpp, thermal_model, seed)
        return
    ### to train
    # add the hyperparameters
    ems_relax = params["ems_relax"]
    learning_rate = params["learning_rate"]
    gamma = params["gamma"]
    update_frequency = params["update_frequency"]
    warm_start = params["warm_start"]
    snr = params["snr"]
    stds = params["stds"]
    if "rcmodel" in thermal_model and ems_relax == "fixed_bin":
        raise Warning("RC model must be used with QP relaxation or SS relaxation. This run configuration was skipped.")
        return
    dnn.train(warm_start, nb_epochs_max, loss_metric, learning_rate, gamma, update_frequency, medoids, medoid_labels,
                   hpp, thermal_model, ems_relax, patience_max=15, seed=seed, snr=snr, std_w=stds, std_b=stds)

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
                     params_df[["loss_metric", "ems_relax", "seed", "stds", "test_only"]]],
                    axis=1).to_string())

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
    filter_bool1 = [not (not p["test_only"] and (p["thermal_model"] == "nn" and p["hpp"]["nb_neurons"] == 10 and p["ems_relax"] in ["fixed_bin", "ss"])) for p in params_all]

    ### Automatic filtering based on the summary file
    if not os.path.exists(summary_filepath):
        # use a print rather than a warning because warning print is not displayed at the right place
        print("\033[31mWARNING: Summary file does not exist. No filtering with summary file will be applied.\033[0m")
        filter_bool2 = [True] * len(params_all)
    else:
        # read the summary file
        summary_df = pd.read_excel(summary_filepath, index_col=0)
        # Keep only successful runs
        summary_df = summary_df.loc[list(map(lambda x: not math.isnan(x), summary_df["Test Loss"]))]
        # Filter out the parameters that have been already run successfully
        summary_diclist = summary_df.to_dict(orient="records")
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
                    one_param_set["stds"] == summary_row["Std w"] and one_param_set["stds"] == summary_row["Std b"])
                # Check the learning rate
                summary_initial_lr = np.round(summary_row["Learning rate"] / (summary_row["gamma"] ** (summary_nb_epochs+1)), decimals=8)
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

        filter_bool2 = [not any(list(map(match_summary_and_params, summary_diclist, repeat(p)))) for p in
                       params_all]

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


def parallel_dfl_training(task_id):
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
    Parallel(n_jobs=-1, verbose=10, backend="loky")(delayed(safe_process_task)(param) for param in params)

    # For debugging, parallelism is to be avoided.
    # for param in params:
    #     print(f"\n\n***Processing parameters: {param}***")
    #     process_task(param)

def paper_parameter_sets(n_tasks):
    """
    This function selects the run configurations to reproduce the results of the paper.
    """
    # get the medoids and labels
    medoids, medoid_labels = get_medoids_and_labels()
    params = [
        # test only RC model
        {'ems_relax': np.nan, 'gamma': np.nan,
         'hpp': {'target': 'Zone Mean Air Temperature(t+1)'}, 'learning_rate': np.nan, 'loss_metric': 'expost+',
         'medoid_labels': medoid_labels, 'medoids': medoids, 'nb_epochs_max': 100, 'seed': 1, 'snr': np.nan,
         'stds': np.nan, 'test_only': True, 'thermal_model': 'rcmodel', 'update_frequency': np.nan,
         'warm_start': np.nan},
        # test only NN1 model
        {'ems_relax': np.nan, 'gamma': np.nan,
         'hpp': {'activation': 'ReLU()', 'nb_inputs': 11, 'nb_layers': 1, 'nb_neurons': 2, 'sparse': False,
                 'target': 'Zone Mean Air Temperature(t+1)'}, 'learning_rate': np.nan, 'loss_metric': 'expost+',
         'medoid_labels': medoid_labels, 'medoids': medoids, 'nb_epochs_max': 100, 'seed': 0, 'snr': np.nan,
         'stds': np.nan, 'test_only': True, 'thermal_model': 'nn', 'update_frequency': np.nan, 'warm_start': np.nan},
        # test only NN2 model
        {'ems_relax': np.nan, 'gamma': np.nan,
         'hpp': {'activation': 'ReLU()', 'nb_inputs': 11, 'nb_layers': 1, 'nb_neurons': 5, 'sparse': False,
                 'target': 'Zone Mean Air Temperature(t+1)'}, 'learning_rate': np.nan, 'loss_metric': 'expost+',
         'medoid_labels': medoid_labels, 'medoids': medoids, 'nb_epochs_max': 100, 'seed': 0, 'snr': np.nan,
         'stds': np.nan, 'test_only': True, 'thermal_model': 'nn', 'update_frequency': np.nan, 'warm_start': np.nan},
        # test only NN3 model
        {'ems_relax': np.nan, 'gamma': np.nan,
         'hpp': {'activation': 'ReLU()', 'nb_inputs': 11, 'nb_layers': 1, 'nb_neurons': 10, 'sparse': False,
                 'target': 'Zone Mean Air Temperature(t+1)'}, 'learning_rate': np.nan, 'loss_metric': 'expost+',
         'medoid_labels': medoid_labels, 'medoids': medoids, 'nb_epochs_max': 100, 'seed': 4, 'snr': np.nan,
         'stds': np.nan, 'test_only': True, 'thermal_model': 'nn', 'update_frequency': np.nan, 'warm_start': np.nan},
        # RC model with QP relaxation
        {'ems_relax': 'qp', 'gamma': 0.98,
         'hpp': {'target': 'Zone Mean Air Temperature(t+1)'}, 'learning_rate': 1e-2, 'loss_metric': 'hierarchical_weighted_mae',
         'medoid_labels': medoid_labels, 'medoids': medoids, 'nb_epochs_max': 100, 'seed': 1, 'snr': 0,
         'stds': 0, 'test_only': False, 'thermal_model': 'rcmodel', 'update_frequency': 1, 'warm_start': "True"},
        # RC model with SS relaxation
        {'ems_relax': 'ss', 'gamma': 0.98,
         'hpp': {'target': 'Zone Mean Air Temperature(t+1)'}, 'learning_rate': 2e-2, 'loss_metric': 'expost+',
         'medoid_labels': medoid_labels, 'medoids': medoids, 'nb_epochs_max': 100, 'seed': 1, 'snr': 0,
         'stds': 1, 'test_only': False, 'thermal_model': 'rcmodel', 'update_frequency': 1, 'warm_start': "True"},
        # NN1 model with QP relaxation
        {'ems_relax': 'qp', 'gamma': 0.98,
         'hpp': {'activation': 'ReLU()', 'nb_inputs': 11, 'nb_layers': 1, 'nb_neurons': 2, 'sparse': False,
                 'target': 'Zone Mean Air Temperature(t+1)'}, 'learning_rate': 1e-3, 'loss_metric': 'hierarchical_weighted_mae',
         'medoid_labels': medoid_labels, 'medoids': medoids, 'nb_epochs_max': 100, 'seed': 4, 'snr': 0,
         'stds': 0, 'test_only': False, 'thermal_model': 'nn', 'update_frequency': 1, 'warm_start': "True"},
        # NN1 model with FB relaxation
        {'ems_relax': 'fixed_bin', 'gamma': 0.98,
         'hpp': {'activation': 'ReLU()', 'nb_inputs': 11, 'nb_layers': 1, 'nb_neurons': 2, 'sparse': False,
                 'target': 'Zone Mean Air Temperature(t+1)'}, 'learning_rate': 1e-3, 'loss_metric': 'hierarchical_weighted_mae',
         'medoid_labels': medoid_labels, 'medoids': medoids, 'nb_epochs_max': 100, 'seed': 4, 'snr': 0,
         'stds': 0, 'test_only': False, 'thermal_model': 'nn', 'update_frequency': 1, 'warm_start': "True"},
        # NN1 model with SS relaxation
        {'ems_relax': 'ss', 'gamma': 0.98,
         'hpp': {'activation': 'ReLU()', 'nb_inputs': 11, 'nb_layers': 1, 'nb_neurons': 2, 'sparse': False,
                 'target': 'Zone Mean Air Temperature(t+1)'}, 'learning_rate': 1e-3, 'loss_metric': 'expost+',
         'medoid_labels': medoid_labels, 'medoids': medoids, 'nb_epochs_max': 100, 'seed': 2, 'snr': 0,
         'stds': 0.05, 'test_only': False, 'thermal_model': 'nn', 'update_frequency': 1, 'warm_start': "True"},
        # NN2 model with QP relaxation
        {'ems_relax': 'qp', 'gamma': 0.98,
         'hpp': {'activation': 'ReLU()', 'nb_inputs': 11, 'nb_layers': 1, 'nb_neurons': 5, 'sparse': False,
                 'target': 'Zone Mean Air Temperature(t+1)'}, 'learning_rate': 1e-3, 'loss_metric': 'hierarchical_weighted_mae',
         'medoid_labels': medoid_labels, 'medoids': medoids, 'nb_epochs_max': 100, 'seed': 1, 'snr': 0,
         'stds': 0, 'test_only': False, 'thermal_model': 'nn', 'update_frequency': 1, 'warm_start': "True"},
        # NN2 model with FB relaxation
        {'ems_relax': 'fixed_bin', 'gamma': 0.98,
         'hpp': {'activation': 'ReLU()', 'nb_inputs': 11, 'nb_layers': 1, 'nb_neurons': 5, 'sparse': False,
                 'target': 'Zone Mean Air Temperature(t+1)'}, 'learning_rate': 1e-3, 'loss_metric': 'hierarchical_weighted_mae',
         'medoid_labels': medoid_labels, 'medoids': medoids, 'nb_epochs_max': 100, 'seed': 0, 'snr': 0,
         'stds': 0, 'test_only': False, 'thermal_model': 'nn', 'update_frequency': 1, 'warm_start': "True"},
        # NN2 model with SS relaxation
        {'ems_relax': 'ss', 'gamma': 0.98,
         'hpp': {'activation': 'ReLU()', 'nb_inputs': 11, 'nb_layers': 1, 'nb_neurons': 5, 'sparse': False,
                 'target': 'Zone Mean Air Temperature(t+1)'}, 'learning_rate': 1e-3, 'loss_metric': 'expost+',
         'medoid_labels': medoid_labels, 'medoids': medoids, 'nb_epochs_max': 100, 'seed': 2, 'snr': 0,
         'stds': 0.01, 'test_only': False, 'thermal_model': 'nn', 'update_frequency': 1, 'warm_start': "True"},
        # NN3 model with QP relaxation
        {'ems_relax': 'qp', 'gamma': 0.98,
         'hpp': {'activation': 'ReLU()', 'nb_inputs': 11, 'nb_layers': 1, 'nb_neurons': 10, 'sparse': False,
                 'target': 'Zone Mean Air Temperature(t+1)'}, 'learning_rate': 1e-3, 'loss_metric': 'hierarchical_weighted_mae',
         'medoid_labels': medoid_labels, 'medoids': medoids, 'nb_epochs_max': 100, 'seed': 1, 'snr': 0,
         'stds': 0, 'test_only': False, 'thermal_model': 'nn', 'update_frequency': 1, 'warm_start': "True"},
    ]
    # Split the parameters into chunks for each task and save them
    split_params(params, n_tasks)



