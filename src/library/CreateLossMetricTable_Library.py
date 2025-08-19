import math
from collections.abc import Callable
import numpy as np
import os
import pandas as pd
import pickle
import re
import sys
import time

def get_paths(results_folderpath):
    # Get the paths to the losses
    train_path = os.path.join(results_folderpath, "losses.pkl")
    test_hot_path = os.path.join(results_folderpath, "TestHotYear/test_losses.pkl")
    # Get the paths to the costs
    cost_path = os.path.join(results_folderpath, "ems.pkl")
    test_hot_cost_path = os.path.join(results_folderpath, "TestHotYear/ems.pkl")
    # Return the paths with hot year if they exist or not
    test_hot_flag = True if os.path.exists(test_hot_path) else False  # test if there is a test hot year
    if not test_hot_flag:
        test_hot_path = None
        test_hot_cost_path = None

    return {"loss": train_path, "loss_hot_year": test_hot_path, "cost": cost_path, "cost_hot_year": test_hot_cost_path}

def read_losses(loss_path, test_hot_path, cost_path, test_hot_cost_path, best_epoch):
    losses = {}

    # read the losses for training, validation, test and, if available, test hot year
    with open(loss_path, "rb") as f:
        tmp = pickle.load(f)
        df = pd.DataFrame(tmp["training"].loc[best_epoch].to_list())  # get a series of dict, convert it to a list of dict
        losses["training"] = df.astype(float).mean(axis=0)
        df=pd.DataFrame(tmp["validation"].loc[best_epoch].to_list())  # get a series of dict, convert it to a list of dict
        losses["validation"] = df.astype(float).mean(axis=0)
        df = pd.DataFrame(tmp["test"].loc[0].to_list())  # get a series of dict, convert it to a list of dict
        losses["test"] = df.astype(float).mean(axis=0)
    if test_hot_path is not None:
        with open(test_hot_path, "rb") as f:
            tmp = pickle.load(f)
            losses["test_hot"] = pd.DataFrame(tmp).astype(float).mean(axis=0)
    else:
        losses["test_hot"] = None

    # read the ems costs for training, validation, test and, if available, test hot year
    epc = "expected_power_cost"
    eppc = "ex-post_power_cost"
    losses["all_costs"] = pd.DataFrame(index=[epc, eppc], columns=["training", "validation", "test", "test_hot"])
    lac = losses["all_costs"]
    with open(cost_path, "rb") as f:
        # Because of file structure change, redirect 'MyLibrary' to 'library'
        sys.modules['MyLibrary'] = sys.modules['src.library']
        # RuntimeWarning comes from here because the cost in test column is empty
        tmp = pickle.load(f)
        if tmp["training"].loc[best_epoch].dropna().empty:
            lac.at[epc, "training"] = np.nan
            lac.at[eppc, "training"] = np.nan
        else:
            lac.at[epc, "training"] = tmp["training"].loc[best_epoch].apply(lambda x: x.expected_power_cost).mean()
            lac.at[eppc, "training"] = tmp["training"].loc[best_epoch].apply(lambda x: x.expost_power_cost).mean()
        if tmp["validation"].loc[best_epoch].dropna().empty:
            lac.at[epc, "validation"] = np.nan
            lac.at[eppc, "validation"] = np.nan
        else:
            lac.at[epc, "validation"] = tmp["validation"].loc[best_epoch].apply(lambda x: x.expected_power_cost).mean()
            lac.at[eppc, "validation"] = tmp["validation"].loc[best_epoch].apply(lambda x: x.expost_power_cost).mean()
        lac.at[epc, "test"] = tmp["test"].loc[0].apply(lambda x: x.expected_power_cost).mean()
        lac.at[eppc, "test"] = tmp["test"].loc[0].apply(lambda x: x.expost_power_cost).mean()

    if test_hot_cost_path is not None:
        with open(test_hot_cost_path, "rb") as f:
            tmp = pickle.load(f)
            tmp.columns = ["test_hot"]
            lac["test_hot"] = tmp.applymap(lambda x: np.mean(x[-1]))

    return losses

def merge_costs(cost_df, test_cost_df):
    # replace test cost column by the value for the test set
    cost_df["test"] = test_cost_df["test"]
    return cost_df

def put_costs_in_df(losses, costs):
    for col in costs.columns:
        losses[col] = pd.concat((losses[col], costs[col]), axis=0)
    return losses

def get_header1(row):
    # Determine the header (column name) of the table
    # first level of column name
    if "2neurons" in row["Model carac."]:
        header1 = "NN1"
    elif "5neurons" in row["Model carac."]:
        header1 = "NN2"
    elif "10neurons" in row["Model carac."]:
        header1 = "NN3"
    elif row["Model carac."] == "rcmodel":
        header1 = "RC"
    elif row["Model carac."] == "spatialrcmodel":
        header1 = "SRC"
    else:
        raise ValueError("The model is not recognized.")

    # second level of column name
    if "fixed_bin" in row["Bin. Formulation"]:
        header2 = "FB"
    elif "ss" in row["Bin. Formulation"]:
        header2 = "SS"
    elif "qp" in row["Bin. Formulation"]:
        header2 = "QP"
    else:
        raise ValueError("The formulation is not recognized.")
    # header2 = "" if "RC" in header1 else header2

    return (header1, header2)

def get_header2(row):
    # Determine the header (column name) of the table
    # first level of column name
    if "2neurons" in row["Model carac."]:
        header1 = "NN1"
    elif "5neurons" in row["Model carac."]:
        header1 = "NN2"
    elif "10neurons" in row["Model carac."]:
        header1 = "NN3"
    elif row["Model carac."] == "rcmodel":
        header1 = "RC"
    elif row["Model carac."] == "spatialrcmodel":
        header1 = "SRC"
    else:
        raise ValueError("The model is not recognized.")

    # second level of column name
    header2 = "ITO"

    return (header1, header2)

def match_losses(losses, table, row, get_header):

    header = get_header(row)
    table_col = table[header]

    # fill the column
    table_col.loc["Hierarchical loss"] = losses["hierarchical_weighted_mae"]
    table_col.loc["MAE (kW)"] = losses["mae"]
    table_col.loc["MSE (kW2)"] = losses["mse"]
    table_col.loc["Error mean (kW)"] = losses["error_mu"]
    table_col.loc["Error std (kW)"] = losses["error_std"]
    table_col.loc["Ex-post+ ($)"] = (
            losses["ex-post_power_cost"] + losses["tin_penalty_expost"] + (losses["expected_power_cost"] - losses["ex-post_power_cost"])**2)
    table_col.loc["Expected cost ($)"] = losses["expected_power_cost"]
    table_col.loc["Ex-post cost ($)"] = losses["ex-post_power_cost"]
    table_col.loc["Cost error ($)"] = losses["ex-post_power_cost"] - losses["expected_power_cost"]
    table_col.loc["Temp. Penalty($)"] = losses["tin_penalty_expost"]
    table_col.loc["Nb. Epochs"] = int(row["Nb Epoch"].split('/')[0])
    table_col.loc["Training time"] = row["Training time"]
    table_col.loc["Validation time"] = row["Validation time"]
    table_col.loc["Test time"] = row["Test time"]

def rounding(table):
    for idx, row in table.iterrows():
        if any(row > 100):
            table.loc[idx] = row.astype(int)
        elif any(row > 10):
            table.loc[idx] = row.astype(float).round(1)
        else:
            table.loc[idx] = row.astype(float).round(2)
    return table

def rounding_row(row):
    if any(row > 100):
        new_row = row.loc[row.notna()].astype(int)  # can handle NaN
    elif any(row > 10):
        new_row = row.loc[row.notna()].astype(float).round(1)
    else:
        new_row = row.loc[row.notna()].astype(float).round(2)
    return new_row


def rewrite_training_time(tt):
    # Define the regular expression for the format XXhxxmxxs
    pattern = r'^\d{2}h\d{2}m\d{2}s$'
    if not bool(re.match(pattern, tt)):
        if tt[0] == '0':
            prefix = ""
        else:
            prefix = tt[0] + "d"
        times = tt[7:].split(":")
        return prefix + times[0] + "h" + times[1] + "m" + times[2] + "s"


def retrieve_losses(table1, best_df, paths, get_header):
    for idx, row in best_df.iterrows():
        thermal_model = row['Model carac.']
        if thermal_model == "rcmodel":
            model = "RC"
        elif thermal_model == "spatialrcmodel":
            model = "spatialRC"
        else:
            model = "NotSparse/NN_" + thermal_model[:-5]
        paths["results_folderpath"] = os.path.join(paths["allresults_folderpath"], f"cvxpylayer/{model}/{row['Date']}")
        best_epoch = row["Best Epoch"]
        # seed = row["Seed"]
        # loss_metric = "hierarchical_weighted_mae"

        ### run new tests and save everything in a new folder
        # rt.run_test_set_on_trained_model(results_folderpath, thermal_model, loss_metric, 'rnd', '2006', seed)

        ### Paths
        paths.update(get_paths(paths["results_folderpath"]))

        ### Read the losses
        losses = read_losses(*[paths[k] for k in ["loss", "loss_hot_year", "cost", "cost_hot_year"]], best_epoch)

        ### Gather the costs in 'all_costs' df
        # losses["all_costs"] = merge_costs(losses["cost"], losses["test_cost"])
        # losses["all_costs"] = merge_costs(losses["all_costs"], losses["test_hot_cost"])
        # losses.pop("cost")
        # losses.pop("test_cost")
        # losses.pop("test_hot_cost")

        # Put the cost in the correct df
        losses = put_costs_in_df(losses, losses.pop("all_costs"))

        # save only the test losses in the table
        match_losses(losses["test"], table1, row, get_header)

def get_best(gb_modelandformulation, interesting_df):
    # for each model, extract the best training of each binary formulation
    best = []
    for m in gb_modelandformulation:
        # best.append(interesting_df.loc[m["Test Loss"].idxmin()])
        best.append(interesting_df.loc[m["Test Expost+"].idxmin()])
    best_df = pd.concat(best, axis=0)
    return best_df


def get_best_and_fill_table(gb_modelandformulation, interesting_df, table, paths, get_header: Callable):
    """

    Args:
        gb_modelandformulation: 
        table1: 
        paths: 

    Returns:

    """
    best_df = get_best(gb_modelandformulation, interesting_df)

    ### Fill the result table
    # retrieve losses
    retrieve_losses(table, best_df, paths, get_header)
    ### Save the tables
    # round the results
    # TMP0 = table.iloc[:-2]
    # TMP1 = TMP0._is_view
    # rounding(table.iloc[:-2, :])

    idx = [i for i in table.index if not "time" in i]
    # table.loc[idx] = table.loc[idx].apply(rounding_row, axis=1)
    for i in idx:
        table.loc[i] = rounding_row(table.loc[i])

    # rewrite training time
    # table.loc["Training time"] = table.loc["Training time"].apply(rewrite_training_time)
    table.loc["Training time"] = table.loc["Training time"].apply(lambda x: time.strftime('%H:%M:%S', time.gmtime(x)) if not isinstance(x, str) and not math.isnan(x) else x)
    table.loc["Validation time"] = table.loc["Validation time"].apply(lambda x: time.strftime('%H:%M:%S', time.gmtime(x)) if not isinstance(x, str) and not math.isnan(x) else x)
    table.loc["Test time"] = table.loc["Test time"].apply(lambda x: time.strftime('%H:%M:%S', time.gmtime(x)) if not isinstance(x, str) and not math.isnan(x) else x)
    table.mask(table == '00:00:00', other='-', inplace=True)  # replace NaN with np.nan

    # correct for epoch starting at 0 if there is no NaN
    table.loc["Nb. Epochs"] = table.loc["Nb. Epochs", table.loc["Nb. Epochs"].notna()].where(table.loc["Nb. Epochs"] == 0, table.loc["Nb. Epochs"]+1).astype(int)

    # save to csv and latex
    table.to_csv(paths["savepath"]+".csv", index=True, header=True, na_rep="NaN")
    table.to_latex(paths["savepath"], index=True, escape=True)

    return table

