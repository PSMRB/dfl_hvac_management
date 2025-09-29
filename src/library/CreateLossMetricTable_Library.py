import math
from collections.abc import Callable
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import re
import scienceplots
import sys
import time

# uses scienceplots style formatting
plt.style.use(['grid', 'science', 'ieee'])
# Set the default plt.show to 100 dpi but savefig to 600 to fit IEEE standards
plt.rcParams.update({'figure.dpi': '100', 'savefig.dpi': '600', 'savefig.format': 'pdf'})

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
        # training
        df = tmp["training"].loc[0].apply(pd.Series)  # convert series of dict into a dataframe
        if not df.dropna().empty:  # if there was a training, weighted average of all the metrics
            losses["training"] = df["overall_weight"] @ df
        else:
            losses["training"] = df.astype(float).mean(axis=0)
        # validation
        df = tmp["validation"].loc[0].apply(pd.Series)  # convert series of dict into a dataframe
        if not df.dropna().empty:  # if there was a validation, weighted average all the metrics
            losses["validation"] = df["overall_weight"] @ df
        else:
            losses["validation"] = df.astype(float).mean(axis=0)
        df = tmp["test"].loc[0].apply(pd.Series)  # convert series of dict into a dataframe
        # losses["test"] = df.astype(float).mean(axis=0)
        ow = df["overall_weight"]
        # ow = np.ones(len(df)) / len(df)  # equal weights for all test samples
        losses["test"] = ow @ df
    if test_hot_path is not None:
        with open(test_hot_path, "rb") as f:
            tmp = pickle.load(f)
            df = pd.DataFrame(tmp).astype(float)
            losses["test_hot"] = df["overall_weight"] @ df
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
            lac.at[epc, "training"] = tmp["training"].loc[best_epoch].apply(lambda x: x.expected_power_cost).mul(ow).sum(axis=0)
            lac.at[eppc, "training"] = tmp["training"].loc[best_epoch].apply(lambda x: x.expost_power_cost).mul(ow).sum(axis=0)
        if tmp["validation"].loc[best_epoch].dropna().empty:
            lac.at[epc, "validation"] = np.nan
            lac.at[eppc, "validation"] = np.nan
        else:
            lac.at[epc, "validation"] = tmp["validation"].loc[best_epoch].apply(lambda x: x.expected_power_cost).mul(ow).sum(axis=0)
            lac.at[eppc, "validation"] = tmp["validation"].loc[best_epoch].apply(lambda x: x.expost_power_cost).mul(ow).sum(axis=0)
        lac.at[epc, "test"] = tmp["test"].loc[0].apply(lambda x: x.expected_power_cost).mul(ow).sum(axis=0)
        lac.at[eppc, "test"] = tmp["test"].loc[0].apply(lambda x: x.expost_power_cost).mul(ow).sum(axis=0)

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


def get_header4(row):
    if row["Model carac."] == "1layers_2neurons_each":
        header1 = "NN1"
    elif row["Model carac."] == "1layers_5neurons_each":
        header1 = "NN2"
    else:
        raise ValueError("The model is not recognized.")
    return (header1, row['S'])

def get_header5(row):
    header1 = "Variable" if row['Std requires grad'] else "Constant"
    # for header 2
    if "->" in row["Std w"]:
        ratio = row["Std w"].split("-> ")[1]
        header2 = f"mu/{int(1/float(ratio))}"
    else:
        header2 = str(row['Std w'])
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
    table_col.loc["Ex-post+ ($)"] = losses["expost+"]
    # TMP0 = losses["ex-post_power_cost"] + losses["tin_penalty_expost"] + (losses["expected_power_cost"] - losses["ex-post_power_cost"])**2
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


def get_model_path_from_summary_row(row, paths):
    thermal_model = row['Model carac.']
    if thermal_model == "rcmodel":
        model = "RC"
    elif thermal_model == "spatialrcmodel":
        model = "spatialRC"
    else:
        model = "NotSparse/NN_" + thermal_model[:-5]
    paths["results_folderpath"] = os.path.join(paths["allresults_folderpath"], f"cvxpylayer/{model}/{row['Date']}")


def retrieve_losses(table, best_df, paths, get_header):
    for idx, row in best_df.iterrows():
        get_model_path_from_summary_row(row, paths)
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
        match_losses(losses["test"], table, row, get_header)

def get_best(gb_modelandformulation, interesting_df):
    # for each model, extract the best training of each binary formulation
    best = []
    for m in gb_modelandformulation:
        # best.append(interesting_df.loc[m["Test Loss"].idxmin()])
        # best.append(interesting_df.loc[m["Test Expost+"].idxmin()])
        best.append(interesting_df.loc[m["Best Val Loss"].idxmin()])
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
    idx = [i for i in table.index if not "time" in i]
    for i in idx:
        table.loc[i] = rounding_row(table.loc[i])

    # rewrite training time
    table.loc["Training time"] = table.loc["Training time"].apply(lambda x: time.strftime('%H:%M:%S', time.gmtime(x)) if not isinstance(x, str) and not math.isnan(x) else x)
    table.loc["Validation time"] = table.loc["Validation time"].apply(lambda x: time.strftime('%H:%M:%S', time.gmtime(x)) if not isinstance(x, str) and not math.isnan(x) else x)
    table.loc["Test time"] = table.loc["Test time"].apply(lambda x: time.strftime('%H:%M:%S', time.gmtime(x)) if not isinstance(x, str) and not math.isnan(x) else x)
    table.mask(table == '00:00:00', other='-', inplace=True)  # replace NaN with np.nan

    # correct for epoch starting at 0 if there is no NaN
    table.loc["Nb. Epochs"] = table.loc["Nb. Epochs", table.loc["Nb. Epochs"].notna()].where(table.loc["Nb. Epochs"] == 0, table.loc["Nb. Epochs"]+1).astype(int)

    # save to csv and latex
    table.to_csv(paths["savepath"] + ".csv", index=True, header=True, na_rep="NaN")
    table.style.format(escape="latex").to_latex(paths["savepath"] + ".tex")

    return table


def plot_std_analysis(feasible_df, paths):
    """

    Args:
        std_df:
        gb_std_mean:
        paths:

    Returns:

    """

    ### pre-processing of the data
    interesting_df = feasible_df.copy()
    # overwrite the number of epochs
    interesting_df["Nb Epoch"] = feasible_df["Nb Epoch"].apply(lambda x: int(x.split("/")[0]))

    def preprocess(model_name, interesting_df):
        # group by the model characteristics (rcmodel, nn architecture, etc.)
        gb_model = interesting_df.groupby("Model carac.")
        # keep only the useful NN architectures
        interesting_df = gb_model.filter(lambda x: x.name in [model_name])
        # remove the tests
        interesting_df = interesting_df.loc[interesting_df["Warm-start"].astype(str) == "True"]
        # keep only the SS formulation
        interesting_df = interesting_df.loc[(interesting_df["Bin. Formulation"] == "ss")]
        # keep only the SS formulation with different S
        interesting_df = interesting_df.loc[(interesting_df["S"] == 1)]
        # keep only ex-post+, nb of epochs, and the times (training, val, test) and infos about std
        interesting_df = interesting_df[["Test Expost+", "Nb Epoch", "Std w", "Std requires grad"]]
        # reindex by std requires grad and std w
        std_df = interesting_df.set_index(["Std requires grad", "Std w"])
        # compute the mean of each group
        gb_std_mean = interesting_df.groupby("Std requires grad").apply(lambda x: x.groupby("Std w").mean())

        return interesting_df, std_df, gb_std_mean

    _, std_df_NN1, std_mean_df_NN1 = preprocess("1layers_2neurons_each", interesting_df)
    _, std_df_NN2, std_mean_df_NN2 = preprocess("1layers_5neurons_each", interesting_df)

    # PLOT
    # get the x axis values according to the std w values: 0.01, 0.05, 0.1, mu/10, mu/2 -> 1, 2, 3, 4, 5
    # if std requires grad is False, then - 0.1, if True, then + 0.1
    x_values = {"0.01": 1, "0.05": 2, "0.1": 3, "1 -> 0.1": 4, "1 -> 0.5": 5}
    x_labels = ["0.01", "0.05", "0.1", "$\mu$/10", "$\mu$/2"]

    # Default parameters
    default_figsize = plt.rcParams["figure.figsize"]  # [width, height]
    plt.rcParams['axes.grid'] = False # turn off grid for all plots

    fig = plt.figure(figsize=(1.2 * default_figsize[0], 1.2 * default_figsize[1]))
    gs = fig.add_gridspec(2, 2, hspace=0.05, wspace=0.05)
    axs = gs.subplots(sharex=True, sharey='row')
    # plot the mean of each group with different markers for std requires grad True or False
    for (std_rg, std_w), group in std_mean_df_NN1.iterrows():
        x = x_values[str(std_w)] + (0.1 if std_rg else -0.1)
        y = group["Test Expost+"]
        axs[0, 0].scatter(x, y, label=r'Variable $\sigma$' if std_rg else r'Constant $\sigma$', marker='.', s=26,
                    color='C1' if std_rg else 'C0')
    # plot the single points with a different marker
    for (std_rg, std_w), group in std_df_NN1.iterrows():
        x = x_values[str(std_w)] + (0.1 if std_rg else -0.1)
        y = group["Test Expost+"]
        axs[0, 0].scatter(x, y, marker='o',  s=12,
                          facecolors='none', edgecolor='C1' if std_rg else 'C0', linewidth=0.7, alpha=0.5)
    axs[0, 0].set_ylabel(r"Ex-post+ (\$)")
    axs[0, 0].set_xticks(list(x_values.values()), x_labels)
    axs[0, 0].tick_params(axis='x',  # changes apply to the x-axis
                    which='minor',  # minor ticks are affected
                    bottom=False,  # ticks along the bottom edge are off
                    top=False,  # ticks along the top edge are off
                    labelbottom=False)  # labels along the bottom edge are off
    axs[0, 0].set_title('NN1')
    # first two legend entries only
    handles, labels = axs[0, 0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    hnd, lbl = list(unique.values()), list(unique.keys())
    leg = fig.legend(hnd, lbl, title="", loc='upper center', bbox_to_anchor=(0.5, -0.0), ncols=2)
    leg.set_frame_on(True)  # turn the frame (box) on
    leg.get_frame().set_edgecolor('black')  # set box edge color
    leg.get_frame().set_linewidth(0.5)  # set box line width

    ### Plot the nb of epochs
    # plot the mean of each group with different markers for std requires grad True or False
    for (std_rg, std_w), group in std_mean_df_NN1.iterrows():
        x = x_values[str(std_w)] + (0.1 if std_rg else -0.1)
        y = group["Nb Epoch"]
        axs[1, 0].scatter(x, y, marker='.', s=26, color='C1' if std_rg else 'C0')
    # plot the single points with a different marker
    for (std_rg, std_w), group in std_df_NN1.iterrows():
        x = x_values[str(std_w)] + (0.1 if std_rg else -0.1)
        y = group["Nb Epoch"]
        axs[1, 0].scatter(x, y, marker='o', facecolors='none', edgecolor='C1' if std_rg else 'C0', linewidth=0.7, s=12,
                    alpha=0.5)
    axs[1, 0].set_ylabel(r"Nb Epochs")
    axs[1, 0].set_xlabel(r"Std $\sigma$")
    axs[1, 0].set_xticks(list(x_values.values()), x_labels)
    axs[1, 0].tick_params(axis='x',  # changes apply to the x-axis
                    which='minor',  # minor ticks are affected
                    bottom=False,  # ticks along the bottom edge are off
                    top=False,  # ticks along the top edge are off
                    labelbottom=False)  # labels along the bottom edge are off

    ### for NN2
    # plot the mean of each group with different markers for std requires grad True or False
    for (std_rg, std_w), group in std_mean_df_NN2.iterrows():
        x = x_values[str(std_w)] + (0.1 if std_rg else -0.1)
        y = group["Test Expost+"]
        axs[0, 1].scatter(x, y, label=f"{'Variable' if std_rg else 'Constant'}", marker='.', s=26,
                    color='C1' if std_rg else 'C0')
    # plot the single points with a different marker
    for (std_rg, std_w), group in std_df_NN2.iterrows():
        x = x_values[str(std_w)] + (0.1 if std_rg else -0.1)
        y = group["Test Expost+"]
        axs[0, 1].scatter(x, y, marker='o', facecolors='none', edgecolor='C1' if std_rg else 'C0', linewidth=0.7, s=12,
                    alpha=0.4)
    axs[0, 1].set_xticks(list(x_values.values()), x_labels)
    axs[0, 1].tick_params(axis='x',  # changes apply to the x-axis
                    which='minor',  # minor ticks are affected
                    bottom=False,  # ticks along the bottom edge are off
                    top=False,  # ticks along the top edge are off
                    labelbottom=False)  # labels along the bottom edge are off
    axs[0, 1].set_title('NN2')

    ### Plot the nb of epochs
    # plot the mean of each group with different markers for std requires grad True or False
    for (std_rg, std_w), group in std_mean_df_NN2.iterrows():
        x = x_values[str(std_w)] + (0.1 if std_rg else -0.1)
        y = group["Nb Epoch"]
        axs[1, 1].scatter(x, y, label=f"{'Variable' if std_rg else 'Constant'}", marker='.', s=26,
                    color='C1' if std_rg else 'C0')
    # plot the single points with a different marker
    for (std_rg, std_w), group in std_df_NN2.iterrows():
        x = x_values[str(std_w)] + (0.1 if std_rg else -0.1)
        y = group["Nb Epoch"]
        axs[1, 1].scatter(x, y, marker='o', facecolors='none', edgecolor='C1' if std_rg else 'C0', linewidth=0.7, s=12,
                    alpha=0.4)
    axs[1, 1].set_xlabel(r"Std $\sigma$")
    axs[1, 1].set_xticks(list(x_values.values()), x_labels)
    axs[1, 1].tick_params(axis='x',  # changes apply to the x-axis
                    which='minor',  # minor ticks are affected
                    bottom=False,  # ticks along the bottom edge are off
                    top=False,  # ticks along the top edge are off
                    labelbottom=False)  # labels along the bottom edge are off


    # save the figure
    fig.savefig(paths["savepath"] + ".pdf", bbox_inches='tight')
    plt.close()


def plot_s_analysis(feasible_df, paths):
    """

    Args:
        std_df:
        gb_std_mean:
        paths:

    Returns:

    """

    ### pre-processing of the data
    interesting_df = feasible_df.copy()
    # overwrite the number of epochs
    interesting_df["Nb Epoch"] = feasible_df["Nb Epoch"].apply(lambda x: int(x.split("/")[0]))

    def preprocess(model_name, interesting_df):
        # group by the model characteristics (rcmodel, nn architecture, etc.)
        gb_model = interesting_df.groupby("Model carac.")
        # keep only the useful NN architectures
        interesting_df = gb_model.filter(lambda x: x.name in [model_name])
        # remove the tests
        interesting_df = interesting_df.loc[interesting_df["Warm-start"].astype(str) == "True"]
        # keep only the SS formulation
        interesting_df = interesting_df.loc[(interesting_df["Bin. Formulation"] == "ss")]
        # keep only the SS formulation with sigma == 0.01
        interesting_df = interesting_df.loc[
            (interesting_df["Std w"] == "0.01") & (~interesting_df["Std requires grad"])]
        # keep only the SS formulation with different S
        interesting_df = interesting_df.loc[(interesting_df["S"] > 0)]
        # keep only ex-post+, nb of epochs, and the times (training, val, test) and S
        interesting_df = interesting_df[["Test Expost+", "Nb Epoch", "Training time",
                                         "S"]]
        # scale the training time to be in hours
        interesting_df["Training time"] = interesting_df["Training time"] / 3600
        # reindex by std requires grad and std w
        std_df = interesting_df.set_index(["S"])
        # compute the mean of each group
        gb_std_mean = interesting_df.groupby("S").mean()

        return interesting_df, std_df, gb_std_mean

    _, s_df_NN1, s_mean_df_NN1 = preprocess("1layers_2neurons_each", interesting_df)
    _, s_df_NN2, s_mean_df_NN2 = preprocess("1layers_5neurons_each", interesting_df)

    # PLOT
    # get the x axis values according to the std w values: 0.01, 0.05, 0.1, mu/10, mu/2 -> 1, 2, 3, 4, 5
    # if std requires grad is False, then - 0.1, if True, then + 0.1
    x_values = {1: 1, 2: 2, 5: 3, 10: 4}
    x_labels = ["1", "2", "5", "10"]

    # Default parameters
    default_figsize = plt.rcParams["figure.figsize"]  # [width, height]
    plt.rcParams['axes.grid'] = False # turn off grid for all plots

    fig = plt.figure(figsize=(1.2 * default_figsize[0], 1.8 * default_figsize[1]))
    gs = fig.add_gridspec(3, 2, hspace=0.05, wspace=0.05)
    axs = gs.subplots(sharex=True, sharey='row')
    # plot the mean of each group with different markers for std requires grad True or False
    for s, group in s_mean_df_NN1.iterrows():
        x = x_values[s]
        y = group["Test Expost+"]
        axs[0, 0].scatter(x, y, marker='.', s=26, color='C0')
    # plot the single points with a different marker
    for s, group in s_df_NN1.iterrows():
        x = x_values[s]
        y = group["Test Expost+"]
        axs[0, 0].scatter(x, y, marker='o',  s=12,
                          facecolors='none', edgecolor='C0', linewidth=0.7, alpha=0.5)
    axs[0, 0].set_ylabel(r"Ex-post+ (\$)")
    axs[0, 0].set_xticks(list(x_values.values()), x_labels)
    axs[0, 0].tick_params(axis='x',  # changes apply to the x-axis
                    which='minor',  # minor ticks are affected
                    bottom=False,  # ticks along the bottom edge are off
                    top=False,  # ticks along the top edge are off
                    labelbottom=False)  # labels along the bottom edge are off
    axs[0, 0].set_title('NN1')

    ### Plot the nb of epochs
    # plot the mean of each group with different markers for std requires grad True or False
    for s, group in s_mean_df_NN1.iterrows():
        x = x_values[s]
        y = group["Nb Epoch"]
        axs[1, 0].scatter(x, y, marker='.', s=26, color='C0')
    # plot the single points with a different marker
    for s, group in s_df_NN1.iterrows():
        x = x_values[s]
        y = group["Nb Epoch"]
        axs[1, 0].scatter(x, y, marker='o', facecolors='none', edgecolor='C0', linewidth=0.7, s=12,
                    alpha=0.5)
    axs[1, 0].set_ylabel(r"Nb Epochs")
    axs[1, 0].set_xticks(list(x_values.values()), x_labels)
    axs[1, 0].tick_params(axis='x',  # changes apply to the x-axis
                    which='minor',  # minor ticks are affected
                    bottom=False,  # ticks along the bottom edge are off
                    top=False,  # ticks along the top edge are off
                    labelbottom=False)  # labels along the bottom edge are off

    ### Plot the training time
    # plot the mean of each group with different markers for std requires grad True or False
    for s, group in s_mean_df_NN1.iterrows():
        x = x_values[s]
        y = group["Training time"]
        axs[2, 0].scatter(x, y, marker='.', s=26, color='C0')
    # plot the single points with a different marker
    for s, group in s_df_NN1.iterrows():
        x = x_values[s]
        y = group["Training time"]
        axs[2, 0].scatter(x, y, marker='o', facecolors='none', edgecolor='C0', linewidth=0.7, s=12,
                    alpha=0.5)
    axs[2, 0].set_ylabel(r"Training time [h]")
    axs[2, 0].set_xlabel(r"$S$")
    axs[2, 0].set_xticks(list(x_values.values()), x_labels)
    axs[2, 0].tick_params(axis='x',  # changes apply to the x-axis
                    which='minor',  # minor ticks are affected
                    bottom=False,  # ticks along the bottom edge are off
                    top=False,  # ticks along the top edge are off
                    labelbottom=False)  # labels along the bottom edge are off

    ### for NN2
    # plot the mean of each group with different markers for std requires grad True or False
    for s, group in s_mean_df_NN2.iterrows():
        x = x_values[s]
        y = group["Test Expost+"]
        axs[0, 1].scatter(x, y, marker='.', s=26, color='C0')
    # plot the single points with a different marker
    for s, group in s_df_NN2.iterrows():
        x = x_values[s]
        y = group["Test Expost+"]
        axs[0, 1].scatter(x, y, marker='o', facecolors='none', edgecolor='C0', linewidth=0.7, s=12, alpha=0.4)
    axs[0, 1].set_xticks(list(x_values.values()), x_labels)
    axs[0, 1].tick_params(axis='x',  # changes apply to the x-axis
                    which='minor',  # minor ticks are affected
                    bottom=False,  # ticks along the bottom edge are off
                    top=False,  # ticks along the top edge are off
                    labelbottom=False)  # labels along the bottom edge are off
    axs[0, 1].set_title('NN2')

    ### Plot the nb of epochs
    # plot the mean of each group with different markers for std requires grad True or False
    for s, group in s_mean_df_NN2.iterrows():
        x = x_values[s]
        y = group["Nb Epoch"]
        axs[1, 1].scatter(x, y, marker='.', s=26, color='C0')
    # plot the single points with a different marker
    for s, group in s_df_NN2.iterrows():
        x = x_values[s]
        y = group["Nb Epoch"]
        axs[1, 1].scatter(x, y, marker='o', facecolors='none', edgecolor='C0', linewidth=0.7, s=12,
                    alpha=0.4)
    axs[1, 1].set_xticks(list(x_values.values()), x_labels)
    axs[1, 1].tick_params(axis='x',  # changes apply to the x-axis
                    which='minor',  # minor ticks are affected
                    bottom=False,  # ticks along the bottom edge are off
                    top=False,  # ticks along the top edge are off
                    labelbottom=False)  # labels along the bottom edge are off

    ### Plot the training time
    # plot the mean of each group with different markers for std requires grad True or False
    for s, group in s_mean_df_NN2.iterrows():
        x = x_values[s]
        y = group["Training time"]
        axs[2, 1].scatter(x, y, marker='.', s=26, color='C0')
    # plot the single points with a different marker
    for s, group in s_df_NN2.iterrows():
        x = x_values[s]
        y = group["Training time"]
        axs[2, 1].scatter(x, y, marker='o', facecolors='none', edgecolor='C0', linewidth=0.7, s=12,
                    alpha=0.5)
    axs[2, 1].set_xlabel(r"$S$")
    axs[2, 1].set_xticks(list(x_values.values()), x_labels)
    axs[2, 1].tick_params(axis='x',  # changes apply to the x-axis
                    which='minor',  # minor ticks are affected
                    bottom=False,  # ticks along the bottom edge are off
                    top=False,  # ticks along the top edge are off
                    labelbottom=False)  # labels along the bottom edge are off


    # save the figure
    fig.savefig(paths["savepath"] + ".pdf", bbox_inches='tight')
    plt.close()
