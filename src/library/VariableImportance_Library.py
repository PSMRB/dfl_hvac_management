from collections import defaultdict

import sklearn

from src.library import Common as c
from src.library.Common import Database, modeltrainandtest
import src.library.Plot as Plot
import numpy as np
import os
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.tree import DecisionTreeRegressor
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform


def prepare_dataset(path_h5, variablestokeep: list):
    # load the training dataset
    data = pd.read_hdf(path_h5, "/df", mode='r')

    # convert datetime string into interpretable values
    M, D, h, _ = c.splitdate(data["Date/Time"])  # minute is not used because it is only zeros
    cum_h = c.getcumulativehours(data["Date/Time"])
    data = data.assign(Month=M, Day=D, Hour=h, Cumulative_Hour=cum_h)

    # Shorten column labels to more interpretable names
    c.renamecolumns(data)

    # Create column to indicate cooling (-1) or heating (+1) (if only ventilation => (0)) or idle (0)
    data["HVAC Mode(t-0)"] = np.where(data["Cooling Coil Electricity Rate [W](t-0)"] > 0, -1, 0)
    data["HVAC Mode(t-0)"] = np.where(data["Heating Coil Electricity Rate [W](t-0)"] > 0, 1, data["HVAC Mode(t-0)"])

    # Delta T with desired temperature, /!\ the setpoint is to be achieved at the end of the timestep
    data["Delta T Setpoint [C](t-0)"] = 0
    for i, r in data.iterrows():
        if r["Zone Mean Air Temperature [C](t-0)"] - r["Zone Thermostat Heating Setpoint Temperature [C](t-0)"] < 0:
            data.at[i, "Delta T Setpoint [C](t-0)"] = r["Zone Mean Air Temperature [C](t-0)"] - r["Zone Thermostat Heating Setpoint Temperature [C](t-0)"]
        elif r["Zone Mean Air Temperature [C](t-0)"] - r["Zone Thermostat Cooling Setpoint Temperature [C](t-0)"] > 0:
            data.at[i, "Delta T Setpoint [C](t-0)"] = r["Zone Mean Air Temperature [C](t-0)"] - r["Zone Thermostat Cooling Setpoint Temperature [C](t-0)"]

    # Differentiate between heating and cooling electricity rates
    data[("Heating Electricity Rate [W](t-0)")] = np.where(data["HVAC Mode(t-0)"] == 1,
                                                           data["Unitary System Electricity Rate [W](t-0)"], 0)
    data[("Cooling Electricity Rate [W](t-0)")] = np.where(data["HVAC Mode(t-0)"] == -1,
                                                           data["Unitary System Electricity Rate [W](t-0)"], 0)
    # Differentiate between heating and cooling electricity rate thanks to sign
    # Ventilation is not considered as electrical power needed for heating or cooling
    data[("Signed Unitary System Electricity Rate [W](t-0)")] = np.where(
        data["HVAC Mode(t-0)"] >= 0, data[("Heating Electricity Rate [W](t-0)")],
        -data[("Cooling Electricity Rate [W](t-0)")])


    # create column for variable to predict: indoor temperature at t+1
    prediction_target = data["Zone Mean Air Temperature [C](t-0)"].shift(-1)
    prediction_target.rename("Zone Mean Air Temperature [C](t+1)", inplace=True)
    data = pd.concat([data, prediction_target], axis=1).dropna()

    # create column for variable to predict: difference of indoor temperature at t+1: \DeltaT = T(t+1) - T(t)
    prediction_target = data["Zone Mean Air Temperature [C](t+1)"] - data["Zone Mean Air Temperature [C](t-0)"]
    prediction_target.rename("Delta Mean Air Temperature [C](t+1)", inplace=True)
    data = pd.concat([data, prediction_target], axis=1)

    # Drop the non-measurable variables and the original Energy+ HVAC electricity consumption and error columns
    # Keep only the measurable variables and the response variable
    data = data.loc[:, variablestokeep]

    return data


def trmodeltrainandtest(db: Database, rnd: int):
    """
    Train and test a tree regressor model
    :param db: the database object with the train and test sets
    :param rnd: random seed
    :return: the tree model.
    """
    model = DecisionTreeRegressor(criterion="squared_error", random_state=rnd, min_impurity_decrease=0.001)
    return modeltrainandtest(db, model)

def modelandimportance(db: Database, save_fig: bool = False, figure_path: str = "", fig_name_suffix: str = ""):
    """
    Train and test a tree regressor model and compute the gini importance and the permutation importance of the
    variables. The variables are ranked by importance and the importance is plotted.
    :param db:
    :param rnd:
    :param save_fig:
    :param figure_path:
    :param fig_name_suffix:
    :return:
    """
    """----------MODELLING----------"""
    model, metrics = trmodeltrainandtest(db, db.seed)
    # Plot the tree
    tree_path = os.path.join(figure_path, "Trees")
    c.createdir(tree_path)
    Plot.treeplot(model, db.predictors, save_fig, tree_path, filename=f"Tree_{fig_name_suffix}")

    """GINI IMPORTANCE"""
    # Compute the Gini importance
    gini_importances = model.feature_importances_  # Get the feature importance
    gini_importance_df = pd.DataFrame(
        {'Feature': db.X_train.columns, 'Gini Decrease': gini_importances})  # df to store the feature importances
    gini_importance_df = gini_importance_df.sort_values('Gini Decrease', ascending=False).reset_index(
        drop=True)  # Sort df by importance in descending order

    # Plot the variables importances as horizontal bar chart
    Plot.importanceplot(gini_importance_df, save_fig, figure_path, filename=f"GiniImportance{fig_name_suffix}")

    """PERMUTATION IMPORTANCE"""
    # Compute the permutation importance
    # print(f"Available metrics: {sklearn.metrics.get_scorer_names()}")  # list of the available metrics
    # metrics on which to base the permutation importance (only the last one is saved)
    scores = ["r2", "neg_mean_absolute_error", "neg_mean_squared_error"]  # "neg_mean_squared_error" is normal mse
    r = permutation_importance(model, db.X_test, db.y_test_true, n_repeats=100, random_state=db.seed, scoring=scores)

    perm_importance_df = pd.DataFrame()
    for s in scores:
        # Print the feature importances
        perm_importance_df = pd.DataFrame(
            {'Feature': db.X_train.columns, 'Permutation Importance Mean': r[s].importances_mean,
             'Permutation Importance Std': r[s].importances_std})  # df to store the feature importances
        perm_importance_df = perm_importance_df.sort_values('Permutation Importance Mean', ascending=False).reset_index(
            drop=True)  # Sort df by importance in descending order (most important first)

        """----------PLOT----------"""
        # Plot the variables importances as horizontal bar chart
        Plot.importanceplot(perm_importance_df, save_fig, figure_path,
                            filename="PermutationImportance" + fig_name_suffix + f"_{s}",
                            error_available=True)

    return gini_importance_df, perm_importance_df


def correlationandclustering(X: pd.DataFrame, save_fig: bool = False, figure_path: str = None, threshold: float = 0):
    """
    Build a correlation matrix and a distance matrix from the correlation matrix. Perform hierarchical clustering on
    the distance matrix and plot the dendrogram. The distance threshold (which determine the number of clusters Nc)
    is set manually by visual analysis of the dendrogram.
    This allows to retain only Nc variables which are supposed to be the least correlated considering Spearman
    coefficient.
    :param X:
    :param save_fig:
    :param figure_path:
    :param threshold:
    :return:
    """
    # Look at the linear correlation between input variables
    pearson = X.corr(method="pearson")
    spearman = X.corr(method="spearman")
    Plot.correlation_coeff(pearson, save_fig, figure_path, filename="PearsonCoefficient")
    Plot.correlation_coeff(spearman, save_fig, figure_path, filename="SpearmanCoefficient")

    # Conversion of the correlation matrix to a distance matrix before performing
    # hierarchical clustering using Ward's linkage.
    distance_matrix = 1 - np.abs(spearman)
    dist_linkage = hierarchy.ward(squareform(distance_matrix))

    # plot dendrogram (tree plot)
    Plot.dendrogramplot(dist_linkage, X.columns, save_fig, figure_path)

    # Indentify key parameters with clustering, threshold set manually by visual analysis of the dendrogram
    if not threshold:
        threshold = input("By analyzing the dendrogram, input a distance \n")
    cluster_ids = hierarchy.fcluster(dist_linkage, threshold, criterion="distance")
    cluster_id_to_feature_ids = defaultdict(list)
    for idx, cluster_id in enumerate(cluster_ids):
        cluster_id_to_feature_ids[cluster_id].append(idx)
    selected_features_index = [v[0] for v in cluster_id_to_feature_ids.values()]
    selected_features = [list(X.columns)[i] for i in selected_features_index]

    return selected_features


def movingaverage(data: pd.DataFrame, lags):
    """
    Function to compute the moving average of each feature passed in data for each lag in lags.
    :param data:
    :param lags: how many previous time steps to consider for the rolling average (inclusive)
    :return: an ndarray containing the moving average variables. It has a shape: n_samples X (n_features*n_lags)
    """
    n_s, n_f = data.shape  # number of samples and features in data_rolling
    rolling_average = np.empty((n_s, n_f * len(lags)))
    rolling_average[:, :] = np.nan
    for l, lag in enumerate(lags):
        # Compute the moving average for the all variables in data_rolling. For a lag of 2 -> mean(X_(t-2), X_(t-1), X_(t))
        for i in range(0, n_s - lag, 1):
            rolling_average[lag + i, l * n_f:(l + 1) * n_f] = np.mean(data.iloc[i:lag + i + 1, :], axis=0)
            # # The indoor temperature at time step t is unknown (so stop in t-1) -> mean(X_(t-2), X_(t-1))
            # rolling_average[lag + i, l * n_f + index_indtemp] = np.mean(data.iloc[i:lag + i, index_indtemp])

    cols = [col.split("(")[0] + f"(m.a. from t-{lag})" for lag in lags for col in list(data.columns)]
    rolling_average_df = pd.DataFrame(data=rolling_average, columns=cols)

    return rolling_average_df.dropna()

def importanceandresultsafterselection(data: pd.DataFrame, resp_name, seed, n_imp_max: int = 5, save_fig: bool = False,
                                       fig_path: str = "", fig_name_suf: str = ""):
    """
    Function to compute the gini and permutation importance of the features in data and to train and test a model
    :param data:
    :param resp_name:
    :param seed:
    :param n_imp_max: the maximum number of important features to consider for the model
    :param save_fig:
    :param fig_path:
    :param fig_name_suf:
    :return:
    """
    db = Database(data, resp_name, shuffle=False)  # Creates all the sets
    gini, perm = modelandimportance(db, save_fig, fig_path, fig_name_suf)
    print(f"Number of predictor features: {len(db.predictors)}")

    # with the "n_imp" most important gini variables
    n_imp = min(n_imp_max, (gini["Gini Decrease"] > 0.005).sum())  # count the number of variables with a importance > threshold and cap the feature selection at n_imp_max
    tmp = gini.loc[:n_imp-1, "Feature"]  # n_imp-1 because .loc is inclusive
    tmp[len(tmp)] = resp_name[0]
    db_gini = Database(data[tmp], resp_name, seed)
    print(f"\nModel with the {n_imp} most gini-important variables")
    trmodeltrainandtest(db_gini, seed)
    print(f"Number of predictor features: {len(db_gini.predictors)}")

    # with the "n_imp" most important permutation variables
    n_imp = min(n_imp_max, perm["Permutation Importance Mean"].count())
    tmp = perm.loc[:n_imp-1, "Feature"]
    tmp[len(tmp)] = resp_name[0]
    db_perm = Database(data[tmp], resp_name, seed)
    print(f"\nModel with the {n_imp} most permutation-important variables")
    trmodeltrainandtest(db_perm, seed)
    print(f"Number of predictor features: {len(db_perm.predictors)}")

    return db, db_gini, db_perm


def variablestokeep(resp_name: str):
    """
    Function to build the list of possible features for variable importance. The list depends on the response variable.
    :param resp_name: the response variable name
    :return:    - possible features: the list of possible feature and the response variable
                - VIsuffix: the suffix to add to the name of the variable importance folder
    """
    # Possible features to predict the indoor temperature
    possible_features = ['Date/Time(t-0)',
                         'Site Outdoor Air Drybulb Temperature [C](t-0)',
                         'Site Outdoor Air Wetbulb Temperature [C](t-0)',
                         'Site Direct Solar Radiation Rate per Area [W/m2](t-0)',
                         'Site Diffuse Solar Radiation Rate per Area [W/m2](t-0)',
                         'Site Direct+Diffuse Solar Radiation Rate per Area [W/m2](t-0)',
                         'Zone Outdoor Air Wind Speed [m/s](t-0)',
                         'Zone Air Relative Humidity [%](t-0)',
                         'Zone Mean Air Temperature [C](t-0)',
                         # 'Zone Thermostat Heating Setpoint Temperature [C](t-0)',
                         # 'Zone Thermostat Cooling Setpoint Temperature [C](t-0)',
                         # "Heating Electricity Rate [W](t-0)",
                         # "Cooling Electricity Rate [W](t-0)",
                         # "Delta T[C] Setpoint [C](t-0)",
                         # 'Unitary System Electricity Rate [W](t-0)',
                         "Signed Unitary System Electricity Rate [W](t-0)",
                         # allows to avoid binaries in the opti
                         # 'HVAC Mode(t-0)',
                         # 'Electricity:Facility [Wh](t-0)',  # because the load in the opti is different
                         'Month(t-0)', 'Day(t-0)', 'Hour(t-0)',
                         'Cumulative_Hour(t-0)',
                         # Only one of the two possible response variables should be included (see below, the if-elif)
                         # 'Zone Mean Air Temperature [C](t+1)',
                         # 'Delta Mean Air Temperature [C](t+1)',
                         ]

    if resp_name == ["Zone Mean Air Temperature [C](t+1)"]:
        possible_features.append("Zone Mean Air Temperature [C](t+1)")
        # Suffix to the variable importance name
        VIsuffix = "Temperature"
    elif resp_name == ["Delta Mean Air Temperature [C](t+1)"]:
        possible_features.append("Delta Mean Air Temperature [C](t+1)")
        # Suffix to the variable importance name
        VIsuffix = "DeltaTemperature"
    else:
        raise ValueError(f"The response variable is not recognized. Got: {resp_name}")

    return possible_features, VIsuffix


def save_results(data, db_perm, folder_path_save, filename_prefix):
    # Save the variables with a significant importance and the response variables
    df_tosave = pd.concat(
        [data.loc[:, "Date/Time(t-0)"].dt.tz_localize(None),  # Excel cannot save datetime with timezone
         db_perm.X, db_perm.y_true], axis=1)
    df_tosave.set_index("Date/Time(t-0)", inplace=True, drop=True)

    if not os.path.exists(folder_path_save):
        os.mkdir(folder_path_save)
    # -1 because of the response variable
    nb_var = len(df_tosave.columns) - 1
    df_tosave.to_hdf(os.path.join(folder_path_save, f"{filename_prefix}_{nb_var}var.h5"), "/df", mode='w')
    df_tosave.to_excel(os.path.join(folder_path_save, f"{filename_prefix}_{nb_var}var.xlsx"), header=True, index=True)
    data.set_index("Date/Time(t-0)", drop=True, inplace=True)
    data.to_hdf(os.path.join(folder_path_save, f"{filename_prefix}_df.h5"), "/df",
                           mode='w')  # too big to go to excel