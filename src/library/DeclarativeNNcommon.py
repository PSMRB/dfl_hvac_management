import math
from copy import deepcopy
import datetime
import dill as pickle
import matplotlib.pyplot as plt
from Cython.Compiler.Pythran import is_type
from matplotlib.ticker import MaxNLocator
import numpy as np
import os
import pandas as pd
import torch
import warnings
import src.library.Common as c


def weighted_mse_loss(input, target, weight):
    """
    All inputs are torch.Tensors
    :param input:
    :param target:
    :param weight:
    :return:
    """
    return (weight * (input - target) ** 2).sum() / weight.sum()


def weighted_mae_loss(input, target, weight):
    """
    All inputs are torch.Tensors
    :param input:
    :param target:
    :param weight:
    :return:
    """
    return (weight * torch.abs(input - target)).sum() / weight.sum()


def hierarchical_loss(input, target, loss):
    """
    Function to compute the loss at each floor and each building.
    :param zone_assets: list of ZoneAsset objects
    :return: torch.Tensor
    """
    floor_losses = {}
    # floor loss (only one floor for Small Office)
    for fl in ["ZN"]:
        input_stack = torch.stack([v for k, v in input.items() if fl in k], dim=0)
        target_stack = torch.stack([v for k, v in target.items() if fl in k], dim=0)
        input_floor = torch.sum(input_stack, dim=0)
        target_floor = torch.sum(target_stack, dim=0)
        loss_floor = loss(input_floor, target_floor)
        floor_losses[fl] = loss_floor
    # building loss
    input_bldg = torch.sum(torch.stack(list(input.values()), dim=0), dim=0)
    target_bldg = torch.sum(torch.stack(list(target.values()), dim=0), dim=0)
    loss_bldg = loss(input_bldg, target_bldg)

    return floor_losses, loss_bldg


def compute_losses(ems, solution):
    """
    Compute the losses for the given solution.
    Args:
        ems:
        solution: a dictionary with variable names as keys and variable values (at the optimum) as values

    Returns:

    """
    # Compute the loss
    mseloss = torch.nn.MSELoss(reduction='mean')
    maeloss = torch.nn.L1Loss(reduction='mean')
    losses = {"mae": [], "mse": [], "mae(P)+mse(T)":[], "weighted_mae": [], "weighted_mse": [], "mae_reg": [], "mse_reg": [],
              "weighted_mse_reg": [],
              "error_mu": [], "error_std": [], "mse+mae": [], "mse+weighted_mae": [], "mse+expostcost": [],
              "tin_deviation": []}
    expost_cost, prices_with_demand_charge = ems.compute_power_cost("expost")
    prices_with_demand_charge_ts = torch.tensor(
        prices_with_demand_charge, requires_grad=False, dtype=torch.float)
    expected_dic, expost_dic = {}, {}
    for b in ems.bldg_assets:
        cnt = 0
        for z in b.zone_assets:
            if z.controlled:
                # Ex-post hvac power in kW
                p_hvac_expected = torch.Tensor(solution.get(f"p_hvac_{z.name}"))
                p_hvac_expost = torch.Tensor(
                    z.expost_results["P_hvac"].resample('1H').sum().values[:-1]) / 1000
                # p_hvac_expost_debug = b.expost_results.loc[:, (z.name, "P_hvac")].resample(f"1H").sum() / 1000
                expected_dic[z.name] = p_hvac_expected
                expost_dic[z.name] = p_hvac_expost
                t_in_expected = torch.Tensor(solution.get(f"t_in_{z.name}"))
                t_in_expost = torch.Tensor(z.expost_results["Tin"].resample('1H').mean().values)
                regularization = (1e-3 * prices_with_demand_charge_ts * p_hvac_expected ** 2).sum()
                losses["mse"].append(mseloss(p_hvac_expost, p_hvac_expected))
                # + mseloss(t_in_expected, t_in_expost))
                losses["mae"].append(maeloss(p_hvac_expected, p_hvac_expost))
                losses["mae(P)+mse(T)"].append(maeloss(p_hvac_expected, p_hvac_expost)
                                                   + mseloss(t_in_expected, t_in_expost))
                losses["weighted_mae"].append(weighted_mae_loss(p_hvac_expected, p_hvac_expost,
                                                                prices_with_demand_charge_ts))
                losses["weighted_mse"].append(weighted_mse_loss(p_hvac_expected, p_hvac_expost,
                                                                prices_with_demand_charge_ts))
                losses["mse_reg"].append(mseloss(p_hvac_expected, p_hvac_expost) + regularization)
                losses["mae_reg"].append(maeloss(p_hvac_expected, p_hvac_expost, ) + regularization)
                losses["weighted_mse_reg"].append(weighted_mse_loss(p_hvac_expected, p_hvac_expost,
                                                                    prices_with_demand_charge_ts)
                                                  + regularization)
                losses["error_mu"].append(torch.mean(p_hvac_expected - p_hvac_expost))
                losses["error_std"].append(torch.std(p_hvac_expected - p_hvac_expost))
                losses["mse+mae"].append(losses["mse"][-1] + losses["mae"][-1])
                losses["mse+weighted_mae"].append(losses["mse"][-1] + losses["weighted_mae"][-1])
                losses["mse+expostcost"].append(losses["mse"][-1] + expost_cost)
                # Warning: not a differentiable loss
                losses["tin_deviation"].append(maeloss(t_in_expost, torch.Tensor(z.Ttgt.loc[ems.ts_ems].values)))
                cnt += 1
    average_losses = {k: torch.mean(torch.stack(v, dim=0)) for k, v in losses.items()}
    floor_losses_mae, bldg_loss_mae = hierarchical_loss(expected_dic, expost_dic,
                                                        lambda x, y: weighted_mae_loss(x, y,
                                                                                       prices_with_demand_charge_ts))
    floor_losses_mse, bldg_loss_mse = hierarchical_loss(expected_dic, expost_dic,
                                                        lambda x, y: weighted_mse_loss(x, y,
                                                                                       prices_with_demand_charge_ts))
    floor_losses_mae = torch.mean(torch.stack(list(floor_losses_mae.values()), dim=0))
    floor_losses_mse = torch.mean(torch.stack(list(floor_losses_mse.values()), dim=0))
    average_losses["hierarchical_weighted_mae"] = losses["weighted_mae"][
                                                      -1] + 5 * floor_losses_mae + 15 * bldg_loss_mae
    average_losses["hierarchical_weighted_mse"] = losses["weighted_mse"][
                                                      -1] + 5 * floor_losses_mse + 15 * bldg_loss_mse
    print(f"MSE: {average_losses['mse']:.2f} - MAE: {average_losses['mae']:.2f} - "
          f"Weighted MSE: {average_losses['weighted_mse']:.2f}")
    return average_losses


def training_loss(base_loss, medoid_weight: float):
    """
    Build the training loss based on the losses computed by the losses function

    :param base_loss: a loss value (scalar) (that probably comes from the dictionary
                        returned by losses function)
    :param medoid_weight: the weight of the medoid in the clustering.
                            => This loss is valid only for a per sample computation
    :return: the training loss
    """
    return medoid_weight * base_loss


def get_losses(medoid_weight, loss_metric, ems, solution):
    """
    Wrapper around compute_losses to add the medoid weight and the temperature penalty.
    compute_losses returns only losses that can serve for backpropagation.
    Args:
        medoid_weight:
        loss_metric:
        ems:
        solution:

    Returns:

    """
    mse = torch.nn.MSELoss(reduction='mean')

    l = compute_losses(ems, solution)
    l["overall_weight"] = torch.tensor(medoid_weight)
    l["expost"] = torch.tensor(ems.expost_obj_val)
    l["expost+"] = torch.tensor(ems.expost_obj_val) + mse(torch.tensor(ems.expected_power_cost),
                                                          torch.tensor(ems.expost_power_cost))
    l["tin_mae"] = torch.tensor([b.temperature_mae for b in ems.bldg_assets]).mean()
    l["tin_max_e"] = torch.tensor([b.temperature_max_e for b in ems.bldg_assets]).max()
    l["tin_penalty_expected"] = torch.tensor(ems.expected_temperature_penalty)
    l["tin_penalty_expost"] = torch.tensor(ems.expost_temperature_penalty)
    l["training_loss"] = training_loss(l[loss_metric], l["overall_weight"])
    return l


def get_weighted_average_of_losses(losses):
    """
    For a given epoch, compute the weighted average loss. I.e, it weights the loss of each sample by the
    weight of the medoid it belongs to.
    :param losses: a nested list of dictionaries. Each dictionary contains multiple losses for a given sample.
    :return: average_of_weighted_losses: a dictionary of list. Each value of the dictionary is a list.
    The i element of the list associated to key k is the weighted average of the epoch i for the loss k.
    """
    # transform the list of lists of dictionaries into a list of one dictionary with
    # each dictionary value that is a list.
    tmp = [c.agg_itr_of_dict(tl_b, lambda x: x) for tl_b in losses]
    # get the weighted average (as a list of dict)
    tmp2 = [c.map_dict(dic, lambda x: np.average(x, weights=dic['overall_weight']))
            for dic in tmp]
    # transform tmp2 into a dict of list
    average_of_weighted_losses = c.agg_itr_of_dict(tmp2, lambda x: x)
    return average_of_weighted_losses


def get_max_of_losses(losses):
    """
    For a given epoch, compute the weighted average loss. I.e, it weights the loss of each sample by the
    weight of the medoid it belongs to.
    :param losses: a nested list of dictionaries. Each dictionary contains multiple losses for a given sample.
    :return: max_loss: a dictionary of list. Each value of the dictionary is a list.
    The i element of the list associated to key k is the weighted average of the epoch i for the loss k.
    """
    # transform the list of lists of dictionaries into a list of one dictionary with
    # each dictionary value that is a list.
    tmp = [c.agg_itr_of_dict(tl_b, lambda x: x) for tl_b in losses]
    # get the weighted average (as a list of dict)
    tmp2 = [c.map_dict(dic, lambda x: np.max(x)) for dic in tmp]
    # transform tmp2 into a dict of list
    max_loss = c.agg_itr_of_dict(tmp2, lambda x: x)
    return max_loss


def compute_weighted_average_cost(cost_nested_list, losses_dataset):
    """
    compute the weighted average of the cost for each epoch.
    :param cost_nested_list: a list of list of costs. Each sublist is a epoch.
    :param losses_dataset: the dictionary storing all the losses for validation or training.
            E.g.: losses["train"] or losses["validation"]
    :return: a list of the weighted average of the cost for each epoch. (one cost per epoch)
    """
    # remove none and nan values from cost_nested_list and losses_dataset
    cost_nested_list = [[c for c in l if (c is not None and not np.isnan(c) and not math.isnan(c))]
                        for l in cost_nested_list]
    losses_dataset = [[l for l in l2 if (l is not None and not isinstance(l, float))]
                        for l2 in losses_dataset]
    weighted_average_cost = []
    # for each epoch, get the list of costs and the list of weights (one weight and one cost per sample)
    for pc_l, dic_l in zip(cost_nested_list, losses_dataset):
        # go from a list of dictionaries to a dictionary of lists
        l_dic = c.agg_itr_of_dict(dic_l, lambda x: x)
        # convert to array with one dimension (to hedge against the case where
        # there is only one sample/epoch)
        ow_np = np.array(l_dic["overall_weight"], ndmin=1)
        # save
        weighted_average_cost.append(np.average(pc_l, weights=ow_np))
    return weighted_average_cost


def normal_distribution(mean, std):
    """
    Create a normal distribution, sample from it.
    Returns the sample and the distribution.
    """
    # Create the normal distribution for each parameter (ensure std > 0)
    param_distrib = torch.distributions.normal.Normal(mean, torch.clamp(std, min=1e-6))
    # Sample from the distribution
    param_sample = param_distrib.sample()
    # Return the sample and the distribution
    return param_sample.detach(), param_distrib


def recursive_normal_distribution(param, ds, dd, std):
    """
    goes to the leaf of the nested dictionary to create and sample the distribution around the parameter.

    :param param: the (nested) dictionary with the parameters
    :param ds: the dictionary to store the sampled parameters
    :param dd: the dictionary to store the distributions
    """

    if isinstance(std, (int, float)):
        for k, v in param.items():
            if isinstance(v, dict):
                # If the value is a dictionary, recursively sample the distribution
                ds[k], dd[k] = {}, {}
                recursive_normal_distribution(v, ds[k], dd[k], std)
            else:
                # If the value is a tensor, sample the distribution
                ds[k], dd[k] = normal_distribution(v, std)
    else:  # we assume std has the same structure as param
        for k, v in param.items():
            if isinstance(v, dict):
                # If the value is a dictionary, recursively sample the distribution
                ds[k], dd[k] = {}, {}
                recursive_normal_distribution(v, ds[k], dd[k], std[k])
            else:
                # If the value is a tensor, sample the distribution
                ds[k], dd[k] = normal_distribution(v, std[k])
    return ds, dd


def wrapper_normal_distribution(param_nn, std):
    """
    For all the parameters, create a normal distribution around the parameter
    return:
        Two tuples with the same structure as param_nn (weights_dic, biases_dic).
        The first one contains the samples, the second one the distributions.
    """
    sd_l = []
    for u, s in zip(param_nn, std):
        # get parameter samples and distributions
        ps, pd = recursive_normal_distribution(u, {}, {}, s)
        sd_l.append((ps, pd))
    return tuple(sd_l)


def map_weights_and_biaises(f, weights, biases):
    """
    Clone the weights and biases of the DNN and apply the function f to the leaf values of each of them.
    :param weights:
    :param biases:
    :return:
    """
    new_w, new_b = deepcopy(weights), deepcopy(biases)
    for bn, w, b in zip(new_w.keys(), new_w.values(), new_b.values()):
        new_w[bn] = c.map_dict(w, f)
        new_b[bn] = c.map_dict(b, f)
    return new_w, new_b


def save_results(summary_filepath, cluster_name, thermodynamics_model_name, ems):
    # Append those results to the summary file
    if os.path.exists(summary_filepath):
        summary_df = pd.read_excel(summary_filepath, index_col=0, header=0)
    else:
        summary_df = pd.DataFrame(
            columns=['Date', 'Cluster', 'Model Name',
                     'Model carac.', 'N_ts', 'Solving time', 'MIP gap', 'Termination condition',
                     'N_cont_var', 'N_int_var', 'N_bin_var', 'N_constraints', 'Electricity Cost'])

    summary_df = pd.concat(
        [summary_df, pd.DataFrame({'Date': datetime.datetime.now(),
                                   'Cluster': cluster_name,
                                   'Model Name': thermodynamics_model_name,
                                   'Model carac.': ems.bldg_assets[0].ml_model_doc,
                                   'N_ts': ems.nb_ts_ems,
                                   'Solving time': ems.solver_status['solving_time'],
                                   'MIP gap': ems.solver_status['mip_gap'],
                                   'Termination condition': ems.solver_status['termination_condition'],
                                   'Electricity Cost': ems.solver_status['obj_val'],
                                   }, index=[summary_df.shape[0]])
         ], axis=0)
    summary_df.to_excel(summary_filepath, index=True, header=True)


def paths_dic(initial_model_folderpath):
    """
    Function to return the dictionary of paths to the files and folders used in the project.
    Returns: paths

    """
    # the path to the summary file to save the results
    cvxpylayer_results_summary_filepath = os.path.normpath("data/SmallOffice/Models/cvxpylayer/"
                                                           "ResultsSummaryCVXPYlayer2.xlsx")
    # create the folder to save the results
    output_folderpath = os.path.abspath(
        "data/SmallOffice/Output-6zones_ASHRAE901_OfficeSmall_STD2022_Denver_2006-2020")
    idf_filepath = os.path.abspath("data/SmallOffice/6zones_ASHRAE901_OfficeSmall_STD2022_Denver.idf")
    epw_filepath = os.path.abspath("data/SmallOffice/USA_CO_Denver.Intl.AP.725650_US.Normals.2006-2020.epw")
    training_data_filepath = os.path.join(output_folderpath, "Datasets/OutputVariablesTraining.h5")
    model_folderpath = os.path.join(output_folderpath, "../Models")
    cvxpylayer_models_folderpath = os.path.join(model_folderpath, "cvxpylayer")
    # figure_folderpath = os.path.abspath(f"../BuildingModel/Figures/Optimization/{now}")
    # c.createdir(figure_folderpath, up=2)
    c.createdir(os.path.dirname(cvxpylayer_results_summary_filepath), up=1)
    c.createdir(cvxpylayer_models_folderpath, up=0)
    
    return {"initial_model_folderpath": initial_model_folderpath,
            "cvxpylayer_results_summary_filepath": cvxpylayer_results_summary_filepath,
            "cvxpylayer_models_folderpath": cvxpylayer_models_folderpath,
            "model_folderpath": model_folderpath,
            "idf_filepath": idf_filepath,
            "epw_filepath": epw_filepath,
            "training_data_filepath": training_data_filepath,
            "output_folderpath": output_folderpath}


def logs(columns, index):
    """
    Create the dataframes and dictionaries to log the weights, biases, losses and ems during training.
    """
    # dataframe for the weights, biases. They change only during training.
    weights_df = pd.DataFrame(columns=columns, index=index)
    biases_df = pd.DataFrame(columns=columns, index=index)
    # dict of dataframe for the losses. Backprop only during training but computed during training, validation and test
    losses_dic = {
        "training": pd.DataFrame(columns=columns, index=index),
        "validation": pd.DataFrame(columns=columns, index=index),
        "test": pd.DataFrame(columns=columns, index=range(1))}
    # dict to record the ems
    ems_dic = {
            "training": pd.DataFrame(columns=columns, index=index),
            "validation": pd.DataFrame(columns=columns, index=index),
            "test": pd.DataFrame(columns=columns, index=range(1))}
    return weights_df, biases_df, losses_dic, ems_dic


def get_attr_to_list(ems_df, f):
    """
    For each element of a DataFrame, apply the function f and return a list of the results.
    Perfect, when the dataframe is made of object, to extract an attribute of each object and return a
     (nested) list of the results.
    """
    return ems_df.dropna().applymap(f).values.tolist()


def get_avg_cost(ems_df, losses_df, f):
    tmp = get_attr_to_list(ems_df, f)
    # remove the infeasible ems (to prevent mismatch when ECOS bugs)
    ems_df_feasible = get_attr_to_list(ems_df, lambda x: x.feasible)
    tmp = [[tt for tt, eff in zip(t, ef) if eff] for t, ef in zip(tmp, ems_df_feasible)]
    return compute_weighted_average_cost(tmp, losses_df)


def epoch_plots(ems_dic, losses_df, loss_metric, paths):
    """
    Make all the plots after each epoch.

    Args:


    """
    medoid_names = ems_dic["training"].columns
    # remove the row associated with epochs not computed yet
    losses = c.map_dict(losses_df, lambda x: x.dropna(how="all").values.tolist())
    # Remove the nan element within already computed epochs
    losses_no_nan = deepcopy(losses)
    for k, v in losses_no_nan.items():
        if len(v) == 0:
            continue
        losses_no_nan[k] = [[x for x in in_l if not isinstance(x, float)] for in_l in v]
    # get the losses weighted by medoid weights
    average_weighted_train_losses = get_weighted_average_of_losses(losses_no_nan["training"])
    average_weighted_val_losses = get_weighted_average_of_losses(losses_no_nan["validation"])
    max_train_losses = get_max_of_losses(losses_no_nan["training"])
    max_val_losses = get_max_of_losses(losses_no_nan["validation"])

    # Plot the mae and max error on the temperature command
    plt.figure(figsize=(3.3, 2.8))

    train_temp_mae_np = average_weighted_train_losses["tin_mae"]
    val_temp_mae_np = average_weighted_val_losses["tin_mae"]
    train_temp_max_e_np = max_train_losses["tin_max_e"]
    val_temp_max_e_np = max_val_losses["tin_max_e"]
    nb_epochs = len(losses["training"])
    x = range(nb_epochs)

    plt.plot(x, train_temp_mae_np, label="Train. MAE", ls='-', c="black")
    plt.plot(x, val_temp_mae_np, label="Val. MAE", ls='-', c="red")
    plt.plot(x, train_temp_max_e_np, label="Train. Max error",
             ls="--", c="black")
    plt.plot(x, val_temp_max_e_np, label="Val. Max error",
             ls="--", c="red")
    plt.legend(bbox_to_anchor=(0.5, 1.25), loc='upper center', ncols=2)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel("Epoch")
    plt.ylabel("Temperature Error (°C)")
    plt.tight_layout()
    plt.savefig(os.path.join(paths["cvxpylayer_model_save_folderpath"], "TemperatureErrors"))
    # plt.show()

    # plot the base training losses for each scenario (no overall weight)
    # Create a cycler from the Tab20 colormap
    plt.figure(figsize=(4, 2.5))
    tab20_cycler = plt.cycler("color", plt.cm.tab20.colors)
    plt.gca().set_prop_cycle(tab20_cycler)
    # get the loss metric used for training for each sample as a flat array
    train_baselosses_np = np.array(c.pad_nested_lists(
        [c.agg_itr_of_dict(tl_b, lambda x: x)[loss_metric] for tl_b in losses_no_nan["training"]], how=np.mean))
    for j in range(train_baselosses_np.shape[1]):
        plt.plot(range(train_baselosses_np.shape[0]), train_baselosses_np[:, j], label=medoid_names[j])
    plt.legend(bbox_to_anchor=(1.03, 1), loc='upper left')
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel("Epoch")
    plt.ylabel("Base Loss")
    plt.tight_layout()
    plt.savefig(os.path.join(paths["cvxpylayer_model_save_folderpath"], "BaseLossCurves"))
    # plt.show()

    # plot the training losses for each scenario considering the overall weight
    # Create a cycler from the Tab20 colormap
    plt.figure(figsize=(4, 2.5))
    tab20_cycler = plt.cycler("color", plt.cm.tab20.colors)
    plt.gca().set_prop_cycle(tab20_cycler)
    # get the loss metric used for training for each sample as a flat array
    train_losses_np = np.array(c.pad_nested_lists(
        [c.agg_itr_of_dict(tl_b, lambda x: x)["training_loss"] for tl_b in losses_no_nan["training"]], how=np.mean))
    for j in range(train_losses_np.shape[1]):
        plt.plot(range(train_losses_np.shape[0]), train_losses_np[:, j], label=medoid_names[j])
    plt.legend(bbox_to_anchor=(1.03, 1), loc='upper left')
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel("Epoch")
    plt.ylabel("Weighted Loss")
    plt.tight_layout()
    plt.savefig(os.path.join(paths["cvxpylayer_model_save_folderpath"], "LossCurves"))
    # plt.show()

    # plot the mean training and valid losses for each epoch
    plt.figure()
    # mean loss metric for each batch of the validation and training set
    # weighted average loss metric for each batch of the validation and training set
    mean_train_losses_np = average_weighted_train_losses[loss_metric]
    mean_val_losses_np = average_weighted_val_losses[loss_metric]
    plt.plot(x, mean_train_losses_np, label="Train.")
    plt.plot(x, mean_val_losses_np, label="Val.")
    plt.legend()
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig(os.path.join(paths["cvxpylayer_model_save_folderpath"], "AverageLossCurves"))
    # plt.show()

    # plot the average obj value expected and expost
    plt.figure()

    train_weighted_average_expected_cost = get_avg_cost(ems_dic["training"], losses_no_nan["training"], lambda x: x.expected_power_cost)
    train_weighted_average_expost_cost = get_avg_cost(ems_dic["training"], losses_no_nan["training"], lambda x: x.expost_power_cost)
    val_weighted_average_expected_cost = get_avg_cost(ems_dic["validation"], losses_no_nan["validation"], lambda x: x.expected_power_cost)
    val_weighted_average_expost_cost = get_avg_cost(ems_dic["validation"], losses_no_nan["validation"], lambda x: x.expost_power_cost)

    plt.plot(x, train_weighted_average_expected_cost, label="Train. Expected", ls="--", c="k")
    plt.plot(x, train_weighted_average_expost_cost, label="Train. Ex-post", ls="-", c="k")
    plt.plot(x, val_weighted_average_expected_cost, label="Val. Expected", ls="--", c="r")
    plt.plot(x, val_weighted_average_expost_cost, label="Val. Ex-post", ls="-", c="r")
    # plt.legend(bbox_to_anchor=(1.03, 1), loc='upper left')
    plt.legend(loc='upper right')
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel("Epoch")
    plt.ylabel("Average energy cost (€)")
    plt.tight_layout()
    plt.savefig(os.path.join(paths["cvxpylayer_model_save_folderpath"], "AverageEnergyCost"))
    # plt.show()

    # Plot the WEIGHTED average and std of errors between expected and ex-post powers
    # compute the weighted average of the mu and std
    error_mu_t = average_weighted_train_losses["error_mu"]
    error_mu_v = average_weighted_val_losses["error_mu"]
    error_std_t = average_weighted_train_losses["error_std"]
    error_std_v = average_weighted_val_losses["error_std"]

    plt.figure()
    plt.errorbar(x, error_mu_t, yerr=error_std_t, capsize=4, label="Train.",
                 ls="--", c="k")
    plt.errorbar(x, error_mu_v, yerr=error_std_v, capsize=4, label="Val.",
                 ls="--", c="r")
    plt.legend(loc='upper right')  # bbox_to_anchor=(1.03, 1), loc='upper left'
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel("Epoch")
    plt.ylabel("HVAC Power Error (kW)")
    plt.tight_layout()
    plt.savefig(os.path.join(paths["cvxpylayer_model_save_folderpath"], "PowerAverageError"))
    # plt.show()

    plt.figure()
    plt.plot(range(len(losses["training"])),
             average_weighted_train_losses["mae"], label="Train.",
             ls="-", c="k")
    plt.plot(range(len(losses["validation"])),
             average_weighted_val_losses["mae"], label="Val.",
             ls="-", c="r")
    plt.legend(loc='upper right')  # bbox_to_anchor=(1.03, 1), loc='upper left'
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel("Epoch")
    plt.ylabel("HVAC Power MAE (kW)")
    plt.tight_layout()
    plt.savefig(os.path.join(paths["cvxpylayer_model_save_folderpath"], "PowerMAE"))
    # plt.show()

    # plot number of infeasible samples
    bar_width = 0.35
    fsbl_train = get_attr_to_list(ems_dic["training"], lambda x: x.feasible)
    fsbl_val = get_attr_to_list(ems_dic["validation"], lambda x: x.feasible)
    train_infeasible = np.sum(np.reshape(~np.array(fsbl_train), (-1, len(medoid_names))), axis=1)
    val_infeasible = np.sum(np.reshape(~np.array(fsbl_val), (-1, len(medoid_names))), axis=1)

    plt.figure()
    x = np.array(x)
    plt.bar(x - bar_width / 2, train_infeasible, bar_width, label='Train.', color='black')
    plt.bar(x + bar_width / 2, val_infeasible, bar_width, label='Validation', color='red')
    plt.legend(loc='upper right')
    plt.xlabel("Epoch")
    plt.ylabel("Nb. Infeasible Samples")
    plt.tight_layout()
    plt.savefig(os.path.join(paths["cvxpylayer_model_save_folderpath"], "NumberInfeasibleSamples"))
    # plt.show()

    plt.close('all')

    return mean_val_losses_np


def epoch_save(paths, ems_dic, losses_dic, weights_df, biases_df):
    # Save the parameters and the loss values
    with open(os.path.join(paths["cvxpylayer_model_save_folderpath"], "weights.pkl"), "wb") as f:
        pickle.dump(weights_df, f)
    with open(os.path.join(paths["cvxpylayer_model_save_folderpath"], "biases.pkl"), "wb") as f:
        pickle.dump(biases_df, f)
    with open(os.path.join(paths["cvxpylayer_model_save_folderpath"], "losses.pkl"), "wb") as f:
        pickle.dump(losses_dic, f)
    with open(os.path.join(paths["cvxpylayer_model_save_folderpath"], "ems.pkl"), "wb") as f:
        pickle.dump(ems_dic, f)
    # Read data from the files to alleviate memory burden (saving remove some attributes from ems)
    with open(os.path.join(paths["cvxpylayer_model_save_folderpath"], "ems.pkl"), "rb") as f:
        ems_dic = pickle.load(f)

    return ems_dic


def summary_save(dnn, hyperparameters, ems_dic, losses_dic, best_epoch, now, stopwatch_dic, seed, snr, warm_start,
                         ems_relaxation_for_convex_opti, epoch, nb_epochs_max, loss_metric, update_frequency,
                 std_w, std_b, std_requires_grad, s):
    # If testing has been performed
    if not losses_dic["test"].dropna().empty:
        test_loss = losses_dic["test"].loc[0].apply(lambda x: x["training_loss"]).sum().numpy()
        test_expost = losses_dic["test"].loc[0].apply(lambda x: x["expost+"] * x["overall_weight"]).sum().numpy()
        test_mae = losses_dic["test"].loc[0].apply(lambda x: x["mae"] * x["overall_weight"]).sum().numpy()
        test_mse = losses_dic["test"].loc[0].apply(lambda x: x["mse"] * x["overall_weight"]).sum().numpy()
    else:  # no test has been performed
        test_loss, test_expost, test_mae, test_mse = np.nan, np.nan, np.nan, np.nan
    # If testing has been performed
    if not ems_dic["test"].dropna().empty:
        ow = losses_dic["test"].loc[0].apply(lambda x: x["overall_weight"].detach().numpy())
        test_mip = ems_dic["test"].loc[0].apply(lambda x: x.solver_status["mip_gap"]).mean()
        test_infsbl = ems_dic["test"].loc[0].apply(lambda x: not x.feasible).sum()
        test_expected_cost = ems_dic["test"].loc[0].apply(lambda x: x.expected_power_cost).mul(ow).sum()
        test_expost_cost = ems_dic["test"].loc[0].apply(lambda x: x.expost_power_cost).mul(ow).sum()
        test_tin_penalty = ems_dic["test"].loc[0].apply(lambda x: x.expost_temperature_penalty).mul(ow).sum()
        ems = ems_dic["test"].iat[0, 0]
    else:  # no test has been performed
        test_mip, test_infsbl, test_expected_cost, test_expost_cost, test_tin_penalty = np.nan, np.nan, np.nan, np.nan, np.nan
        ems = ems_dic["training"].iat[0, 0]

    # if there was NO training and validation (testing only)
    if ems_dic["training"].dropna().empty and ems_dic["validation"].dropna().empty:
        new_result = {
            'Learning rate': 0,
            'gamma': 0,
            'Update Frequency': 0,
            'Std w': 0,
            'Std b': 0,
            'Training time': 0,
            'Validation time': 0,
            'Train Infsbl': 0,
            'Val Infsbl': 0,
            'Best Train Loss': 0,
            'Best Val Loss': 0,
        }
    else:  # there was training and validation
        new_result = {
            'Learning rate': dnn.optimizer.param_groups[0]["lr"],
            'gamma': dnn.scheduler.gamma,
            'Update Frequency': update_frequency,
            'Training time': str(stopwatch_dic["training"].get_elapsed_time()).split('.')[0],
            'Validation time': str(stopwatch_dic["validation"].get_elapsed_time()).split('.')[0],
            'Train Infsbl': ems_dic["training"].loc[best_epoch].apply(lambda x: not x.feasible).sum(),
            'Val Infsbl': ems_dic["validation"].loc[best_epoch].apply(lambda x: not x.feasible).sum(),
            'Best Train Loss': losses_dic["training"].loc[best_epoch].apply(training_loss_if_feasible).sum().numpy(),
            'Best Val Loss': losses_dic["validation"].loc[best_epoch].apply(training_loss_if_feasible).sum().numpy(),
        }

    # In all cases, save the following
    model_carac = f"{hyperparameters['nb_layers']}layers_{hyperparameters['nb_neurons']}neurons_each" \
        if ems.thermal_model == "nn" else ems.thermal_model
    new_result.update({'Date': now,
                  'Clusters': ", ".join(md for md in ems_dic["training"].columns.to_list()),
                  'Bin. Formulation': ems_relaxation_for_convex_opti,
                  'Model carac.': model_carac,
                  'Seed': seed,
                  'SNR': snr,
                  'Warm-start': warm_start,
                  'Std w': str(std_w) if not callable(std_w) else f"1 -> {std_w(1)}",
                  'Std b': str(std_b) if not callable(std_b) else f"1 -> {std_b(1)}",
                  'Std requires grad': std_requires_grad,
                  'S': s,
                  'N_ts': ems.nb_ts_ems,
                  'Nb Epoch': f"{epoch}/{nb_epochs_max}",
                  'Best Epoch': best_epoch,
                  'Loss metric': loss_metric,
                  'Test time': str(stopwatch_dic["test"].get_elapsed_time()).split('.')[0],
                  'Test MIP gap': test_mip,
                  'Test Infsbl': test_infsbl,
                  'Test Loss': test_loss,
                  'Test Expost+': test_expost,
                  'Test MAE': test_mae,
                  'Test MSE': test_mse,
                  'Test Expected Cost': test_expected_cost,
                  'Test Expost Cost': test_expost_cost,
                  'Test Cost Misestimation': test_expost_cost - test_expected_cost,
                  'Test Tin Penalty': test_tin_penalty,
                  })

    with open(os.path.join(dnn.paths["cvxpylayer_model_save_folderpath"], "summary.pkl"), "wb") as f:
        pickle.dump(new_result, f)


def save_stopwatches(folder_path, stopwatch_dic):
    # save the stopwatch_dic
    with open(os.path.join(folder_path, "stopwatch.pkl"), "wb") as f:
        pickle.dump(stopwatch_dic, f)


def any_sample_feasible(ems_s):
    """
    Check that at least one sample is feasible in the row of the pandas.Series
    :param ems_s: Series of the ems
    :return: True if any sample is feasible, False otherwise
    """
    return ems_s.apply(lambda x: x.feasible).any()


def training_loss_if_feasible(x):
    if isinstance(x, dict):
        y = x["training_loss"]
    else:
        y = 0
    return y


def print_losses(epoch, nb_epochs_max, losses_dic, loss_metric):
    text = f"\n\n*** Epoch {epoch}/{nb_epochs_max - 1} - "
    # Count infeasible samples in training and validation
    infeasible_train = losses_dic["training"].loc[epoch].apply(lambda x: isinstance(x, float)).sum()
    infeasible_val = losses_dic["validation"].loc[epoch].apply(lambda x: isinstance(x, float)).sum()
    # write text
    text += f"Nb infeasible training: {infeasible_train} - "
    if  infeasible_train == 0:
        text += f"Training Loss: {losses_dic['training'].loc[epoch].apply(lambda x: x[loss_metric]).mean():.2f} - "
    text += f"Nb infeasible validation: {infeasible_val}"
    if infeasible_val == 0:
        text += f" - Validation Loss: {losses_dic['validation'].loc[epoch].apply(lambda x: x[loss_metric]).mean():.2f}"
    text += "***"

    print(text)


