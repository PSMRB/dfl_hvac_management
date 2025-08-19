"""
Gather all the scripts which are necessary only once. Typically, scripts to obtain a constant which do not change.
"""
import os.path
from cycler import cycler
import matplotlib as mpl
import matplotlib.pyplot as plt
import library.Common as c
import numpy as np
import pandas as pd
import scienceplots
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler


# Get the number of hours from the previous months
def nhourstillmonth():
    ndays = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    return [sum(ndays[:i]) * 24 for i in range(12)]


# Determine the extreme days of the year where the setpoints can be met
def getextremedays():
    """
    Function to get the extreme days of the year where the setpoints can be met.
    If the setpoints cannot be physically met, then the day is not considered.
    :return:
    """

    # load the data
    year_range_str = "2006-2020"
    filepath = (f"data/SmallOffice/Output-6zones_ASHRAE901_OfficeSmall_STD2022_Denver_{year_range_str}/Datasets/"
                f"OutputVariablesTraining.h5")
    data = pd.read_hdf(filepath, "/df")

    # Discard the sizing days
    nb_sizing_ts = 0
    data = data.iloc[nb_sizing_ts:, :]
    # Discard January 1st 2017 (only 23h) and  last time-step (first hour of January 1st of next year)
    # <=> discard all the first of january
    data = data.loc[data.index.date != pd.Timestamp(year=2017, month=1, day=1).date(), :]

    # filter the days where setpoints can be met
    Tin = "Zone Mean Air Temperature,Core_ZN"
    coolingstpt = "Zone Thermostat Cooling Setpoint Temperature,Core_ZN"
    heatingstpt = "Zone Thermostat Heating Setpoint Temperature,Core_ZN"
    data["hourly_mask"] = data[Tin].between(data[heatingstpt], data[coolingstpt])

    # Check the condition for each day and create a boolean mask
    daily_data = data.groupby(data.index.date)
    filtered_data = daily_data.filter(lambda x: ((x[heatingstpt] - 0.2 <= x[Tin])  # Tin in the
                                                 & (x[Tin] <= x[coolingstpt] + 0.2)).all())   # bounds
    nb_valid_days = filtered_data.shape[0] / 24
    has_nan = filtered_data.isna().values.any()

    # get the extreme days
    Tamb = "Site Outdoor Air Drybulb Temperature,ENVIRONMENT"
    sunshine = "Site Direct+Diffuse Solar Radiation Rate per Area, ENVIRONMENT"
    weight = 0.8
    Tamb_normalized = StandardScaler().fit_transform(filtered_data[Tamb].values.reshape(-1, 1))
    sunshine_normalized = StandardScaler().fit_transform(filtered_data[sunshine].values.reshape(-1, 1))
    filtered_data["criterium"] = weight * Tamb_normalized + (1 - weight) * sunshine_normalized
    daily_filtered_data = filtered_data.groupby(filtered_data.index.date)
    idmax = daily_filtered_data["criterium"].mean().idxmax()
    idmin = daily_filtered_data["criterium"].mean().idxmin()
    idvarmax = daily_filtered_data["criterium"].var().idxmax()
    idextremedays = pd.DataFrame([idmin, idmax, idvarmax],
                                 columns=["Date"], index=["CritMin", "CritMax", "VarMax"])

    # extract the extreme days
    extremedays = []
    for i, idx in idextremedays.iterrows():
        rowlab = pd.date_range(f"{idx[0].strftime('%d/%m/%Y')} 00:00:00",
                               f"{idx[0].strftime('%d/%m/%Y')} 23:00:00", freq="1h")
        extremedays.append(data.loc[rowlab, :])

    # save the extreme days date
    # folderpath = os.path.normpath("../optimization_parameters/TypicalDays")
    # for i, df in enumerate(extremedays):
    #     namesuffix = ["Min", "Max"][i]
    #     df.to_csv(os.path.join(folderpath, f"ExtremeDay_{namesuffix}.csv"), header=True, na_rep="NaN")
    #     # Excel cannot same time zone aware datetime objects
    #     df.insert(0, "Date/Time", df.index)
    #     df["Date/Time"] = df["Date/Time"].apply(lambda x: x.strftime("%Y-%m-%d %H:%M:%S"))
    #     df.reset_index(drop=True, inplace=True)
    #     df.to_excel(os.path.join(folderpath, f"ExtremeDay_{namesuffix}.xlsx"), header=True, na_rep="NaN")
    #     df.to_hdf(os.path.join(folderpath, f"ExtremeDay_{namesuffix}.h5"), "/df")

    return idextremedays, extremedays, filtered_data


def k_medoids_with_fixed_medoids(X, k, fixed_medoids, max_iter=100, tol=1e-5, nb_restarts=50):
    """
    Compute the k-medoids clustering with some fixed medoids.
    :param X: np.ndarray
            the database of the samples
    :param k: the total number of medoids
    :param fixed_medoids: the medoids that are fixed
    :param max_iter:
    :param tol:
    :return:
    """
    m, n = X.shape
    # Ensure fixed_medoids are part of the dataset
    fixed_medoids_idx = [np.nonzero(np.all(X == medoid, axis=1))[0][0] for medoid in fixed_medoids]
    # Get all the samples that are not fixed medoids
    remaining_idx = np.setdiff1d(range(m), fixed_medoids_idx)
    # save the cost of each iteration
    ttl_distances, labels_l, medoids_l, medoids_idx_l = ([] for _ in range(4))
    for r in range(nb_restarts):
        # sample randomly k - nb of fixed medoids
        rng = np.random.default_rng(seed=r)
        random_medoids_idx = rng.choice(remaining_idx, k-len(fixed_medoids_idx), replace=False)
        # initial medoids are the fixed one and the random ones
        medoids_idx = np.concatenate((fixed_medoids_idx, random_medoids_idx))
        medoids = X[medoids_idx, :]
        labels = np.zeros(m)
        for _ in range(max_iter):
            # Compute distances from data points to medoids, return a m x k matrix
            distances = cdist(X, medoids, 'euclidean')
            # Assign each data point to the closest medoid
            labels = np.argmin(distances, axis=1)
            old_medoids = medoids.copy()
            # Update only the non-fixed medoids
            for i in range(len(fixed_medoids_idx), k):
                cluster_idx = np.nonzero(labels == i)[0]
                if cluster_idx.size > 0:
                    cluster_distances = cdist(X[cluster_idx, :], X[cluster_idx, :], 'euclidean')
                    costs = cluster_distances.sum(axis=1)
                    medoids_idx[i] = cluster_idx[np.argmin(costs)]
            medoids = X[medoids_idx, :]
            # Check for convergence
            if np.all(old_medoids == medoids):
                ttl_distances.append(distances.sum())
                labels_l.append(labels)
                medoids_l.append(medoids)
                medoids_idx_l.append(medoids_idx)
                break

    # return the results of the best run (lowest total distance)
    best_run = np.argmin(ttl_distances)

    print(f"Best run {best_run} - Total distance: {ttl_distances[best_run]}")

    return labels_l[best_run], medoids_l[best_run], medoids_idx_l[best_run]


def typicaldayclustering(idextremedays, extremedays, data=None):
    ################################################################
    # cluster the data over outdoor temperature and solar radiation
    ################################################################

    # load the data
    year_range_str = "2006-2020"
    filepath = (f"data/SmallOffice/Output-6zones_ASHRAE901_OfficeSmall_STD2022_Denver_{year_range_str}/Datasets/"
                f"OutputVariablesTraining.h5")
    data = pd.read_hdf(filepath, "/df")

    # Discard the sizing days
    nb_sizing_ts = 0
    data = data.iloc[nb_sizing_ts:, :]
    # Discard January 1st 2017 (only 23h) and  last time-step (first hour of January 1st of next year)
    # <=> discard all the first of january
    data = data.loc[data.index.date != pd.Timestamp(year=2017, month=1, day=1).date(), :]

    # filter the days where setpoints can be met
    Tin = "Zone Mean Air Temperature,Core_ZN"
    coolingstpt = "Zone Thermostat Cooling Setpoint Temperature,Core_ZN"
    heatingstpt = "Zone Thermostat Heating Setpoint Temperature,Core_ZN"
    data["hourly_mask"] = data[Tin].between(data[heatingstpt], data[coolingstpt])
    # Check the condition for each day and create a boolean mask
    daily_data = data.groupby(data.index.date)
    filtered_data = daily_data.filter(lambda x: ((x[heatingstpt] - 0.2 <= x[Tin])  # Tin in the
                                                 & (x[Tin] <= x[coolingstpt] + 0.2)).all())   # bounds
    # parameters
    nb_clusters = 10
    Tin_weight = 0.8
    Tamb = "Site Outdoor Air Drybulb Temperature,ENVIRONMENT"
    sunshine = "Site Direct+Diffuse Solar Radiation Rate per Area, ENVIRONMENT"
    features = [Tamb, sunshine]

    ### Prepare the data for clustering
    # Because of hour change (summer/winter time), some days have 23 or 25 time-steps. We discard them.
    filtered_data = filtered_data.groupby(pd.Grouper(freq='D')).filter(lambda x: x.shape[0] == 24)
    databyday = filtered_data[features].groupby(pd.Grouper(freq='D'))
    # get the days as horizontal arrays of 48 dimensions (24 time-steps and 2 variables)
    daily_data_series = databyday.apply(lambda x: np.concatenate((x.values[:, 0],
                                                                  x.values[:, 1])).flatten()
                                        ).filter(items=filtered_data.index)
    # pd.Timestamp sets the time to 00:00:00 per default
    pos_idx_extremedays = [daily_data_series.index.get_loc(pd.Timestamp(d)) for d in idextremedays["Date"].values]

    # standardize the data
    kmeans_input = np.array(daily_data_series.tolist()).reshape(-1, 48)
    scaler = StandardScaler()
    kmeans_input = scaler.fit_transform(kmeans_input)

    # weight the temperature and solar radiation
    solar_radiation_weight = 1 - Tin_weight
    # coefficient vector to weight the temperature and solar radiation in an element-wise product
    coeff_vector = np.concatenate((np.full(24, Tin_weight), np.full(24, solar_radiation_weight)),
                                                            axis=0)
    kmeans_input = np.multiply(kmeans_input, coeff_vector)
    # prepare extremedays
    extremedays = np.array(
        [extremedays[i].loc[:, features].to_numpy().T.flatten() for i in range(len(extremedays))])
    extremedays_normalized = scaler.transform(extremedays)
    extremedays_input = np.multiply(extremedays_normalized, coeff_vector)
    # cluster the data
    labels, medoids_out, medoids_idx = k_medoids_with_fixed_medoids(kmeans_input, nb_clusters,
                                                                extremedays_input, max_iter=100, nb_restarts=100)
    medoids = scaler.inverse_transform(np.divide(medoids_out, coeff_vector))

    medoids_dates = daily_data_series.iloc[medoids_idx].index
    medoids_names = list(idextremedays.index) + [f"Med. {i}" for i in range(nb_clusters - idextremedays.shape[0])]
    filtered_data["cluster"] = (np.array(labels, ndmin=2, dtype=np.int32).reshape(-1, 1) *
                       np.ones((1,  24), dtype=np.int32)).flatten()

    # retrieve the medoids
    tmp = filtered_data.loc[:, features].groupby(pd.DatetimeIndex(filtered_data.index.date))
    medoids_days = [tmp.get_group(d) for d in list(map(pd.Timestamp, medoids_dates))]

    # check the descaling took place correctly
    for i, df in enumerate(medoids_days):
        assert np.all(df.values.T.flatten() == medoids[i].round(2))

    ### PLOTS ###

    plot_folderpath = os.path.normpath(f"data/SmallOffice/Figures/TypicalDays_{year_range_str}/"
                                       f"{nb_clusters}clusters")
    c.createdir(plot_folderpath, up=2)

    # uses scienceplots style formatting
    plt.rcdefaults()
    plt.style.use(['ieee'])
    plt.rcParams["savefig.format"] = 'pdf'
    plt.rcParams["savefig.dpi"] = 600
    plt.rcParams["figure.dpi"] = 100
    databyday = filtered_data.groupby(filtered_data.index.date)
    daily_mean_temp = databyday.mean()[features[0]]
    daily_mean_solar = databyday.mean()[features[1]]
    daily_mean_df = pd.DataFrame({features[0]: daily_mean_temp, features[1]: daily_mean_solar,
                                  "cluster": labels})
    cluster_mean_df = daily_mean_df.groupby("cluster").mean()
    cluster_mean_df.columns = [f"Mean {f}" for f in features]
    cluster_std_df = daily_mean_df.groupby("cluster").std()
    cluster_std_df.columns = [f"Std {f}" for f in features]
    # gather the clustered day dates in the dataframe
    bincount = pd.DataFrame(np.bincount(labels), columns=["Nb Samples"])
    cluster_df = pd.concat([pd.Series(medoids_dates, name="Date"), bincount, cluster_mean_df, cluster_std_df],
                           axis=1)
    cluster_df.index = [medoids_names[i] for i in cluster_df.index]
    cluster_df.sort_values("Mean " + features[0], inplace=True)

    # bar plot of the cluster temperature and solar radiation
    fig, ax = plt.subplots(1, 2, figsize=(4.5, 2.5))
    ax = ax.flatten()
    xtick_labels = cluster_df.index
    for f in range(2):
        ax[f].bar(xtick_labels, cluster_df["Mean " + features[f]], color="tab:blue",
                  yerr=cluster_df["Std " + features[f]], ecolor="black")
        ax[f].set_xticklabels(xtick_labels, rotation=45, ha='right')
    ax[0].set_ylabel("Ambient Temperature [°C]")
    ax[1].set_ylabel("Solar Radiation [W/m$^2$]")
    # fig.suptitle("Clusters")
    fig.tight_layout()
    fig.savefig(os.path.join(plot_folderpath, "BarPlotClusters"))
    fig.show()

    # Custom color cycle
    tab20cmap = plt.colormaps.get_cmap('tab20')
    mpl.rcParams['axes.prop_cycle'] = cycler(color=tab20cmap.colors)
    # activate the grid
    plt.style.use(['grid'])

    # plot the clusters
    plt.figure()
    plt.scatter(databyday.mean()[features[0]], databyday.mean()[features[1]], c=databyday.mean()["cluster"],
                cmap='tab20')
    plt.xlabel("Daily Average Ambient Temperature [°C]")
    plt.ylabel("Daily Average Solar Radiation [W/m$^2$]")
    plt.title("Clusters of typical days")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_folderpath, "Clustering"))
    plt.show()

    # plot the medoids
    fig, ax = plt.subplots(2, 1, figsize=(4, 3.5), sharex=True, layout="constrained")
    x = range(24)
    for i, df in enumerate(medoids_days):
        legendlab = ["Min", "Max", "var$^{\\rm max}$"][i] if i <= 2 else f"Med. {i - 3}"
        ax[0].plot(x, df[features[0]], label=legendlab)
        ax[1].plot(x, df[features[1]])
    # Place the legend at the bottom of the figure
    fig.legend(loc='outside lower right', ncol=5, fontsize=7, handlelength=1.5, borderpad=0.5, labelspacing=0.5)
    # leave some space for the legend
    ax[0].set_ylabel("Amb. Temperature [°C]")
    ax[1].set_ylabel("Solar radiation [W/m²]")
    plt.subplots_adjust(bottom=0.2)
    plt.xlim([0, 23])
    plt.xlabel("Hour")
    # plt.tight_layout()  # does not work with plt.subplots_adjust
    plt.savefig(os.path.join(plot_folderpath, "TemperaturesAndRadiations"))
    plt.show()

    cluster_df.columns = ["Date", "Nb Samples", "Mean Tamb", "Mean Sun Rad", "Std Tamb", "Std Sun Rad"]
    # save the clustered days
    folderpath = os.path.normpath(f"../Script/OptimizationParameters/{year_range_str}")
    c.createdir(folderpath, up=2)
    cluster_df.to_csv(os.path.join(folderpath, f"Clusters{nb_clusters}Days.csv"), header=True, index=True,
                          na_rep="NaN")

    # Save the labels and the days in a dataframe
    labels_df = pd.DataFrame(list(map(lambda x: medoids_names[x], labels)), columns=["Cluster"],
                                  index=daily_data_series.index)
    # save the label of the days
    labels_df.to_csv(os.path.join(folderpath, f"Labels{nb_clusters}Days.csv"), header=True, index=True, na_rep="NaN")

    debug_stop = 0


def main_clustering():
    """Run the scripts that are necessary only once."""
    # identify the extreme days where HVAC can reach the setpoints
    cluster_idx, cluster, _ = getextremedays()
    # use the dataset of the feasible days to cluster the typical days
    typicaldayclustering(cluster_idx, cluster)
