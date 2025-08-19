
import datetime
import sys

from eppy.modeleditor import IDF
import numpy as np
import os
from os.path import join, normpath
import pandas as pd
import tempfile


from src.library import Common as c



def smoothing(Pnet, Qnet):
    h = 20
    m = len(Pnet)
    Pnet_pred = np.zeros(m)
    Qnet_pred = np.zeros(m)
    P_cont = np.tile(Pnet,2)
    Q_cont = np.tile(Qnet,2)
    for i in range(m):
        Pnet_pred[i] = sum(P_cont[i:i+h])/(h)
        Qnet_pred[i] = sum(Q_cont[i:i+h])/(h)
    return{"Pnet_pred": Pnet_pred, "Qnet_pred": Qnet_pred}


def setpaths(VIsuffix: str, folder_path: str, gbt_time: str, nn_arch_and_time: str, t_in_model_type: str,
             n_features: int, collinearity: bool):
    # Path to the ML models
    # 2023-09-13 14h23m23 : 100 estimators, 2023-09-13 08h19m56s : 10 estimators, 2023-09-13 13h13m48s : 20 estimators
    gbt_path_tail = f"GradientBoostingTrees/{gbt_time}"
    # NN_1layers_10neurons/2023-09-10 10h07m02s, NN_1layers_5neurons/2023-09-13 08h20m37s, NN_2layers_5neurons/2023-09-10 09h25m57s
    nn_path_tail = f"NN/NotSparse/{nn_arch_and_time}"
    lr_path_tail = f"LinearRegression"
    pbe_path_tail = f"PhysicsBasedEquation"

    # collinearity suffix
    collinearity_suffix = "NoCollinearity" if collinearity == False else ""

    bldg_ml_model_path_tail = {
        "lr": lr_path_tail, "nn": nn_path_tail, "gbt": gbt_path_tail, "pbe": pbe_path_tail}[t_in_model_type]
    bldg_ml_model_path = normpath(
        folder_path + f"/Models/{VIsuffix}Models/{n_features}InputVariables{collinearity_suffix}"
                      f"/{bldg_ml_model_path_tail}")

    return bldg_ml_model_path, bldg_ml_model_path_tail


def setpathsII(folder_path, VIsuffix, season, n_features, collinearity, bldg_ml_model_path_tail_for_results):
    """

    :param folder_path: the path to the folder containing the optimization results in general
    :param VIsuffix:
    :param season:
    :param n_features:
    :param collinearity: bool to indicate if the collinearity removal has been performed or not
    :param bldg_ml_model_path_tail_for_results:
    :return:
    """
    # collinearity suffix
    collinearity_suffix = "NoCollinearity" if collinearity == False else ""

    result_path = normpath(join(folder_path,
                                f'{VIsuffix}/{season}/{n_features}features{collinearity_suffix}'
                                f'/{bldg_ml_model_path_tail_for_results}'))
    figure_path = normpath(join(result_path, "Figures"))
    # path to the summary of the expected results
    summary_path = join(folder_path, 'SummaryExpectedResults.xlsx')
    c.createdir(result_path, up=6)
    c.createdir(figure_path, up=0)
    return result_path, figure_path, summary_path


def v_and_i_analysis(networks_l: list, T: int):
    """
    This function is used to analyze the voltage and current results of the power flow simulations.
    :param networks_l: the list of networks for which the power flow has been run. Each element of the list is a
    different time step.
    :param T: number of time steps
    :return:
        - the deviations of the voltage from the ideal voltage (in pu) for each phase (3)
        - the maximum deviation of the voltage from the ideal voltage (in pu)
        - the statistics of the voltage and current for each phase (mean, std, min, max, under v occurrences,
            over v/i occurrences)
    """

    PF_network_res = networks_l

    # one df per phase with columns labels = bus numbers
    v_df = [pd.DataFrame(columns=PF_network_res[0].res_bus_df['number'])] * 3
    i_df = [pd.DataFrame(columns=range(PF_network_res[0].N_lines))] * 3

    # get index of Va column in the res_bus_df
    Va_index = PF_network_res[0].res_bus_df.columns.get_loc('Va')
    Ia_index = PF_network_res[0].res_lines_df.columns.get_loc('Ia')

    # For each time step, for each phase, extract the voltage and current values
    for t in range(T):
        for p in range(3):
            # voltage
            new_elmt = PF_network_res[t].res_bus_df.iloc[:, Va_index + p].abs()
            new_elmt = new_elmt.to_frame().transpose()
            v_df[p] = pd.concat([v_df[p], new_elmt], axis=0, ignore_index=True)
            # current
            new_elmt = PF_network_res[t].res_lines_df.iloc[:, Ia_index + p].abs()
            new_elmt = new_elmt.to_frame().transpose()
            i_df[p] = pd.concat([i_df[p], new_elmt], axis=0, ignore_index=True)

    # Get the ideal voltage
    ideal_voltage = PF_network_res[0].Vslack_ph

    # Calculate deviations and get the maximum deviation for the color scale
    voltage_deviations_pu = []
    max_deviation_pu = 0
    for p in range(3):
        # Calculate deviations from the ideal voltage in p.u.
        new_elmt = (v_df[p].values - ideal_voltage) / ideal_voltage
        # Get the maximum deviation in terms of absolute value for the image color scale
        max_deviation_pu = np.max([max_deviation_pu, np.max(np.abs(new_elmt))])
        # Append the new element to the list of deviations
        voltage_deviations_pu.append(new_elmt)

    # Voltage and current statistics
    v_and_i_stats = pd.DataFrame(
        columns=['Va', 'Vb', 'Vc', 'Ia', 'Ib', 'Ic'],
        index=['mean', 'std', 'min', 'max', 'under v occurrences', 'over v/i occurrences'])
    for p in range(3):
        v_and_i_stats.iloc[0, p] = np.mean(v_df[p].values)
        v_and_i_stats.iloc[1, p] = np.std(v_df[p].values)
        v_and_i_stats.iloc[2, p] = np.min(v_df[p].values)
        v_and_i_stats.iloc[3, p] = np.max(v_df[p].values)
        v_and_i_stats.iloc[4, p] = np.sum(v_df[p].values < PF_network_res[0].v_abs_min[:, p])
        v_and_i_stats.iloc[5, p] = np.sum(v_df[p].values > PF_network_res[0].v_abs_max[:, p])
        v_and_i_stats.iloc[0, p + 3] = np.mean(i_df[p].values)
        v_and_i_stats.iloc[1, p + 3] = np.std(i_df[p].values)
        v_and_i_stats.iloc[2, p + 3] = np.min(i_df[p].values)
        v_and_i_stats.iloc[3, p + 3] = np.max(i_df[p].values)
        v_and_i_stats.iloc[5, p + 3] = np.sum(i_df[p].values > PF_network_res[0].i_abs_max[:, p])

    return voltage_deviations_pu, max_deviation_pu, v_df, i_df, v_and_i_stats


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
                                   # Pyomo extraction of variable numbers
                                   # 'N_cont_var': ems.solver_status['status'].problem.number_of_continuous_variables,
                                   # 'N_int_var': ems.solver_status['status'].problem.number_of_integer_variables,
                                   # 'N_bin_var': ems.solver_status['status'].problem.number_of_binary_variables,
                                   # 'N_constraints': ems.solver_status['status'].problem.number_of_constraints,
                                   # CVXPY extraction of variable numbers
                                   # 'N_cont_var': ems.solver_status['status'].NumVars -
                                   #               ems.solver_status['status'].NumBinVars,
                                   # 'N_int_var': ems.solver_status['status'].NumIntVars,
                                   # 'N_bin_var': ems.solver_status['status'].NumBinVars,
                                   # 'N_constraints': ems.solver_status['status'].NumConstrs,
                                   'Electricity Cost': ems.solver_status['obj_val'],
                                   # 'Ttl Temperature Deviation': Tin_stats_general['ttl deviation'].sum(),
                                   # 'Ttl Overheating': Tin_stats_general['ttl overheating'].sum(),
                                   # 'Ttl Overcooling': Tin_stats_general['ttl overcooling'].sum(),
                                   # 'Ttl Voltage violations': np.sum([np.where(np.abs(v) > 0.1, 1, 0).sum()
                                   #                                   for v in v_deviations_pu]),
                                   # 'Ttl Voltage Deviation p.u.': np.sum([np.abs(v).sum() for v in v_deviations_pu]),
                                   # 'Ttl Undervoltage p.u.': np.sum(
                                   #     [np.where(v < 0, v, 0).sum() for v in v_deviations_pu]),
                                   # 'Ttl Overvoltage p.u.': np.sum(
                                   #     [np.where(v > 0, v, 0).sum() for v in v_deviations_pu])
                                   }, index=[summary_df.shape[0]])
         ], axis=0)
    summary_df.to_excel(summary_filepath, index=True, header=True)


def modify_idf(idf_filepath, T0, idd_filepath=None):
    #                                          /Applications/EnergyPlus-22-1-0/Energy+.idd
    # version 22-2-0_arm64                     /Applications/EnergyPlus-22-2-0_arm64/Energy+.idd
    # CECI version 22.1.0   /home/users/f/a/favarop/.local/src/EnergyPlus-22.1.0_copy/Products/Energy+.idd
    """
    Modify the IDF file.
    :param idf_filepath: idf file path, the building model input data file
    :param idd_filepath: the file to the EnergyPlus Input Data Dictionary. Must be adapted to the version of EnergyPlus.
    :param T0: a pd.Timestamp object representing the day of interest
    :return:
    """
    if idd_filepath == None:
        energyplus_folderpath = sys.path[0]
        idd_filepath = os.path.join(energyplus_folderpath, "Energy+.idd")

    IDF.setiddname(idd_filepath)
    idf = IDF(idf_filepath)
    runperiod = idf.idfobjects["RunPeriod"][0]
    # Modify IDF file to run only the day of interest and the some days because, E+ runs from 01:00 to 24:00
    # but also to have the building dynamic correct.
    startday = T0 - pd.Timedelta(days=5)
    runperiod.Begin_Month, runperiod.Begin_Day_of_Month, runperiod.End_Month, runperiod.End_Day_of_Month = (
        startday.month, startday.day, T0.month, T0.day)
    # print(runperiod)
    # make sure warm-up days are numerous enough for good computation
    bldg = idf.idfobjects["Building"][0]
    nb_warmup_days = 10
    bldg.Maximum_Number_of_Warmup_Days = nb_warmup_days
    bldg.Minimum_Number_of_Warmup_Days = nb_warmup_days
    # Set the tolerance for the unmet setpoints
    tolerance_unmet_stpt = idf.idfobjects["OutputControl:ReportingTolerances"][0]
    tolerance_unmet_stpt.Tolerance_for_Time_Heating_Setpoint_Not_Met = 0.1
    tolerance_unmet_stpt.Tolerance_for_Time_Cooling_Setpoint_Not_Met = 0.1
    # Modify the HVAC schedule to be available all the time (but not modifying the Design Days)
    compact_schedules = idf.idfobjects["Schedule:Compact"]
    for cs in compact_schedules:
        if "HVACOperationSchd" == cs.Name:
            cs.Field_2 = "For: SummerDesignDay"
            cs.Field_9 = "For: WinterDesignDay"
            cs.Field_12 = "For: AllOtherDays"
            cs.Field_14 = 1.0
    # Modify the AvailabilityManager:NightCycle to reduce the thermostat tolerance to 0.
    # Not really necessary since we have modified the hvac schedule to be available all the time
    all_amnc = idf.idfobjects["AvailabilityManager:NightCycle"]
    for amnc in all_amnc:
        amnc.Thermostat_Tolerance = 0
    # save_filepath = idf_filepath.replace(".idf", "_expost.idf")
    # idf.saveas(save_filepath)
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".idf", delete=False) as tmpfile:
        idf.saveas(tmpfile.name)

    return tmpfile.name




