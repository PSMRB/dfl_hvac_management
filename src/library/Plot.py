from datetime import datetime, timedelta
import matplotlib
from matplotlib import colors
import matplotlib.pyplot as plt
import scienceplots
import numpy as np
import os
from os.path import join, normpath
import pandas as pd
from scipy.cluster import hierarchy
from sklearn.tree import plot_tree
from src.library import Common as c

# uses scienceplots style formatting
plt.style.use(['grid', 'ieee', 'no-latex'])
# Set the default plt.show to 100 dpi but savefig to 600 to fit IEEE standards
plt.rcParams.update({'figure.dpi': '100', 'savefig.dpi': '600', 'savefig.format': 'svg'})


class MidpointNormalize(colors.Normalize):
    """
    Allow to normalize the color bars.
    """
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))



def twodaysplot(container_df):
    """
    Function used in SensibleLoadActuator_v3.py to demonstrate that the actuation is working.
    :param container_df: dataframe which contains the building variables for one year
    :return:
    """
    # plot of two days to show it works
    for lb, ub in [(0, 23), (5016, 5039)]: # index of 2 days
        times = []
        for t in container_df.loc[lb:ub, "Timestep"]:
            hour_str = t[-5:] if t[-5:] != "24:00" else "00:00"
            nb_days = 0 if hour_str != "00:00" else 1  # float(hour_str[:2]) // 24  # the number of day
            hour_delta = datetime.strptime(hour_str, '%H:%M') - datetime.strptime('00:00', '%H:%M')
            hour_float = hour_delta.total_seconds() / timedelta(hours=1).total_seconds()
            times.append(24 * nb_days + hour_float)


        fig, ax = plt.subplots(1)
        ax.plot(times, container_df.loc[lb:ub, c.getcolumnname("Heating Coil Heating Rate", list(container_df.columns))]
                - container_df.loc[lb:ub, c.getcolumnname("Cooling Coil Sensible Cooling Rate", list(container_df.columns))])
        ax.set_ylabel("Heat transfer from coils [W]", color="C0")
        ax.set_xlabel("Hour of the day")

        ax2 = ax.twinx()
        ax2.plot(times, container_df.loc[lb:ub, "Zone Mean Air Temperature,THERMAL ZONE LIVING SPACE"], color="r")
        ax2.set_ylabel("Indoor temperature [°C]", color="r")
        ax2.set_title(f"From timesteps {container_df.loc[lb, 'Timestep']} to {container_df.loc[ub, 'Timestep']}")

        # fig.suptitle(f"For timesteps {lb} to {ub}")

        fig.show()

def yearlyanalysis(building: c.BuildingModel, meters_column_name_new: list,
                   folder_path: str, save_fig: bool = False, filename1: str = "HeatingAndCoolingVsWeatherVariables",
                   filename2: str = "HeatingAndCoolingVsIndoorTemperature"):
    """
    Function to analyse the dataset output by Energy+ when performing the yearly simulation. It is used in
    GenerateDatasets.py. It outputs two plots.

    :param final_file: contains the data of the simulation
    :param meters_column_name_new: list which contains the name of the columns (str) where the electricity consumption
    for cooling and heating is recorded
    :param frequency: the frequency of the simulation (generally, "Hourly")
    :param base_folder_path: the path to the base folder, the Energy+ model we are working with and its setpoint
    confirguration
    :param save_fig: Should the figures be saved: True or False ?
    :return:
    """
    dataset = building.simulation
    meter1_column_name_new = meters_column_name_new[0]
    meter2_column_name_new = meters_column_name_new[1]
    heating_on = (dataset[meter2_column_name_new] > dataset[meter1_column_name_new]) & (
            dataset[meter2_column_name_new] > 0)
    cooling_on = (dataset[meter1_column_name_new] >= dataset[meter2_column_name_new]) & (
            dataset[meter1_column_name_new] > 0)
    idle = ~(heating_on | cooling_on)

    # get the useful colum names
    columns = dataset.columns
    drybulb_temp = c.getcolumnname("Drybulb Temperature", columns)
    ttl_solar_rad = c.getcolumnname("Direct+Diffuse Solar Radiation Rate per Area", columns)
    wind_speed = c.getcolumnname("Wind Speed", columns)
    RH = c.getcolumnname("Relative Humidity", columns)
    bldg_mat = c.getcolumnname("Building Mean Air Temperature", columns)


    # Plot: Electricity vs Weather conditions
    fig, axes = plt.subplots(2, 2, sharey='all', figsize=(5, 4))

    # Subplot 1: Electricity vs Outdoor Temperature
    axes[0, 0].plot(dataset[drybulb_temp][heating_on], dataset[meter2_column_name_new][heating_on], "ro ", alpha=1, ms=1)
    axes[0, 0].plot(dataset[drybulb_temp][cooling_on], dataset[meter1_column_name_new][cooling_on], "bs", alpha=1, ms=1)
    axes[0, 0].plot(dataset[drybulb_temp][idle], dataset[meter1_column_name_new][idle], color="black", marker='^',
                    alpha=1, linestyle="None", ms=1)
    axes[0, 0].set_ylabel("HVAC Electricity[Wh]")
    axes[0, 0].set_xlabel("Outdoor Air Drybulb Temperature [C]")
    # use scientific notation for y-axis if number are out of [10^0, 10^2]
    axes[0, 0].ticklabel_format(axis="y", style="sci", scilimits=(0, 2))
    axes[0, 0].legend(["Heating", "Cooling", "Idle"])

    # Subplot 2: Electricity vs Solar radiation
    axes[0, 1].plot(dataset[ttl_solar_rad][heating_on], dataset[meter2_column_name_new][heating_on], "ro", alpha=1, ms=1)
    axes[0, 1].plot(dataset[ttl_solar_rad][cooling_on], dataset[meter1_column_name_new][cooling_on], "bs", alpha=1, ms=1)
    axes[0, 1].plot(dataset[ttl_solar_rad][idle], dataset[meter1_column_name_new][idle], color="black", marker="^",
                    alpha=1, linestyle="None", ms=1)
    axes[0, 1].set_xlabel("Solar Irradiance [W/m2]")

    # Subplot 3: Electricity vs Wind Speed
    axes[1, 0].plot(dataset[wind_speed][heating_on], dataset[meter2_column_name_new][heating_on], "ro", alpha=1, ms=1)
    axes[1, 0].plot(dataset[wind_speed][cooling_on], dataset[meter1_column_name_new][cooling_on], "bs", alpha=1, ms=1)
    axes[1, 0].plot(dataset[wind_speed][idle], dataset[meter1_column_name_new][idle], color="black", marker="^",
                    alpha=1, linestyle="None", ms=1)
    axes[1, 0].set_ylabel("HVAC Electricity[Wh]")
    axes[1, 0].set_xlabel("Wind Speed [m/s]")

    # Subplot 4: Electricity vs Humidity
    axes[1, 1].plot(dataset[RH][heating_on], dataset[meter2_column_name_new][heating_on], "ro", alpha=1, ms=1)
    axes[1, 1].plot(dataset[RH][cooling_on], dataset[meter1_column_name_new][cooling_on], "bs", alpha=1, ms=1)
    axes[1, 1].plot(dataset[RH][idle], dataset[meter1_column_name_new][idle], color="black", marker="^", alpha=1,
                    linestyle="None", ms=1)
    axes[1, 1].set_xlabel("Relative Humidity [%]")

    fig.tight_layout()
    fig.show()

    # to save in the folder model
    if save_fig:
        figure_directory = os.path.join(folder_path, "Figures/YearlyAnalysis")
        c.createdir(figure_directory, up=1)
        fig.savefig(f"{figure_directory}/{filename1}")

    # Plot2: Electricity vs Indoor Temperature
    fig, axes = plt.subplots(1, 1)

    # Subplot 1: Electricity vs Indoor Temperature
    axes.plot(dataset[bldg_mat][heating_on], dataset[meter2_column_name_new][heating_on], "ro", alpha=1, ms=1)
    axes.plot(dataset[bldg_mat][cooling_on], dataset[meter1_column_name_new][cooling_on], "bs", alpha=1, ms=1)
    axes.plot(dataset[bldg_mat][idle], dataset[meter1_column_name_new][idle], color="black", alpha=1, marker="^",
              linestyle="None", ms=1)
    axes.set_ylabel("HVAC Electricity[Wh]")
    axes.set_xlabel("Building Indoor Mean Air Temperature [°C]")
    axes.set_xlim(16, 28)
    # use scientific notation for y-axis if number are out of [10^0, 10^2]
    axes.ticklabel_format(axis="y", style="sci", scilimits=(0, 2))
    axes.legend(["Heating", "Cooling", "Idle"], loc="upper left")

    fig.tight_layout()
    fig.show()
    if save_fig:
        fig.savefig(f"{figure_directory}/{filename2}")

    # Plot3: Electricity vs Indoor Temperature for each of the 15 controlled zone_names (the three plenum, there is no
    # actuation on them)
    nrows = 1
    ncols = 5
    fig, axes = plt.subplots(nrows, ncols, sharey='col', figsize=(2.2*ncols, 2.2*nrows))
    axes = axes.flatten()
    zmat = c.getcolumnname("Zone Mean Air Temperature", dataset.columns)
    for i, z in building.zones_df.loc[~building.zones_df["is_plenum"]].reset_index(drop=True).iterrows():
        if f"Zone Air System Sensible Heating Rate,{z['name']}" in dataset.columns:
            heating = dataset[f"Zone Air System Sensible Heating Rate,{z['name']}"]
            cooling = dataset[f"Zone Air System Sensible Cooling Rate,{z['name']}"]
            heating_on = (heating > cooling) & (heating > 0)
            cooling_on = (heating <= cooling) & (cooling > 0)
            idle = ~(heating_on | cooling_on)
            axes[i].plot(dataset.loc[heating_on, zmat], heating.loc[heating_on], "ro", alpha=1, ms=1)
            axes[i].plot(dataset.loc[cooling_on, zmat], cooling.loc[cooling_on], "bs", alpha=1, ms=1)
            axes[i].plot(dataset.loc[idle, zmat], heating.loc[idle], color="black", marker="^", alpha=1,
                            linestyle="None", ms=1)
            axes[i].set_title(z["name"])
            if i % 5 == 0:
                axes[i].set_ylabel("HVAC Electricity[Wh]")
            elif i == 13:
                axes[i].set_xlabel("Indoor Mean Air Temperature [°C]")
            # use scientific notation for y-axis if number are out of [10^0, 10^2]
            axes[i].ticklabel_format(axis="y", style="sci", scilimits=(0, 2))

    fig.legend(["Heating", "Cooling", "Idle"], loc="lower center", ncol=3)
    fig.tight_layout()
    plt.show()
    if save_fig:
        fig.savefig(f"{figure_directory}/HeatingAndCoolingVsIndoorTemperaturePerZone")

    # Get the correlation matrices
    df = building.simulation.filter(like=f"Zone Mean Air Temperature", axis=1)
    plt.figure(figsize=(6, 6))
    im = plt.imshow(df.corr())
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(df.shape[1]), labels=[col.split(",")[1] for col in df.columns], rotation=90)
    plt.yticks(range(df.shape[1]), labels=[col.split(",")[1] for col in df.columns])
    # Write the exact value of the coefficients as text annotations.
    for i in range(df.shape[1]):
        for j in range(df.shape[1]):
            plt.text(j, i, np.round(df.corr().iloc[i, j], decimals=2), ha="center", va="center", color="w")
    plt.grid(False)
    plt.title("Zone Mean Air Temperature Correlation Matrix")
    plt.tight_layout()
    if save_fig:
        plt.savefig(os.path.join(figure_directory, "CorrelationMatrix"))
    plt.show()



def importanceplot(importance_df: pd.DataFrame, save_fig: bool = False, folderpath: str = None,
                   filename: str = "ImportancePlot", error_available: bool = False):
    """
    Plot a horizontal bar chart of the importance of the variables. Used in VariableImportance.py.
    Only the 20 most important features are plotted.
    :param folderpath:
    :param save_fig:
    :param error_available:
    :param filename:
    :param importance_df:
    :return:
    """
    font = {'size': 10}
    matplotlib.rc('font', **font)
    n_feature_max = 10
    n_features = importance_df.shape[0]
    n = min(n_features, n_feature_max)
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.barh(importance_df.iloc[:n, 0],  # 'Feature'
            importance_df.iloc[:n, 1])  # 'Importance'
    if error_available:
        ax.barh(importance_df.iloc[:n, 0],  # 'Feature'
                importance_df.iloc[:n, 1],
                xerr=importance_df.iloc[:n, 2]*1.96)  # std*1.96 gives the 95% confidence interval
    ax.invert_yaxis()
    ax.set_xlabel(importance_df.columns[1])  # label of axis x is the name of the column
    ax.set_title('Variable Importance')
    ax.set_xlim(left=0)
    fig.tight_layout(rect=[0, 0, 1, 1])
    fig.show()
    if save_fig:
        fig.savefig(os.path.join(folderpath, f"{filename}"))


def treeplot(model, predictors, save_fig: bool = False, folderpath: str = None, filename: str = "DecisionTree"):
    """
    Plot the decision tree once trained.
    :param model: the tree model
    :param predictors: the name of the features used to predict (as input of th three)
    :param save_fig: Bool if to save figure or not
    :param folderpath: In which folder to save the figure
    :param filename: The name of the file
    :return:
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    plot_tree(model, feature_names=predictors, filled=True, fontsize=None)
    ax.set_title("Decision tree trained on all the features")
    # fig.show()
    if save_fig:
        fig.savefig(os.path.join(folderpath, f"{filename}.svg"))


def distribution(data: pd.Series, nbins: int = 10, save_fig: bool = False, folderpath: str = None,
                 filename: str = "distribution"):
    # Plot the histogram
    plt.hist(data, bins=nbins, edgecolor='black', color='grey')
    plt.xlabel(f'{data.name}')
    plt.ylabel('N° of occurences')
    plt.title('Distribution of Data')
    if save_fig:
        plt.savefig(os.path.join(folderpath, f"{filename}"))
    plt.tight_layout()
    plt.show()

    # Print further analysis of the error
    print(f"Minimum error: {data.min()}")
    print(f"Maximum error: {data.max()}")
    print(f"Mean absolute error (MAE): {data.abs().mean()}")
    print(f"Median error: {data.median()}")
    print(f"Q5 and Q95 error: {data.quantile(0.05)} and {data.quantile(0.95)}")


def correlation_coeff(coeffs_matrix: pd.DataFrame, save_fig: bool = False, folderpath: str = None,
                      filename: str = "CorrelationMatrix"):
    variable_names = list(coeffs_matrix.columns)
    ticks = np.arange(0, coeffs_matrix.shape[1])
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    im = ax.imshow(coeffs_matrix.to_numpy(), cmap="bwr", vmin=-1, vmax=1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(variable_names, rotation="vertical")
    ax.set_yticklabels(variable_names)
    fig.colorbar(im, fraction=0.15, shrink=0.7, anchor=(0, 0.5), location="right")
    ax.set_title(filename)
    fig.tight_layout()
    fig.show()
    if save_fig:
        fig.savefig(os.path.join(folderpath, f"{filename}"))


def dendrogramplot(dist_linkage, col_names, save_fig: bool = False, folderpath: str = None,
                      filename: str = "Dendrogram"):
    fig, ax = plt.subplots(1, 1, figsize=(8, 12))
    dendro = hierarchy.dendrogram(dist_linkage, labels=col_names, ax=ax, leaf_rotation=90, show_contracted=True)
    fig.show()
    if save_fig:
        fig.savefig(os.path.join(folderpath, f"{filename}"))


def tempforecast(y_true_batch, y_hat_batch, batch_num: int = None, save_fig: bool = False, folderpath: str = None,
                      filename: str = "LR_IndTemp_24hforecast"):
    """
    Plot the forecast and actual temperature profile for the batch.
    :param y_true_batch: the target value of the batch (list)
    :param y_hat_batch: the forecasted value of the batch (list)
    :param batch_num: the number of the batch
    :return:
    """
    #  maximum four plots per axis => two vectors per table
    n = min(2, y_true_batch.shape[1])
    fig, ax = plt.subplots(1, 1)
    ax.plot(y_true_batch[:, :n], label='Target')
    ax.plot(y_hat_batch[:, :n], label='Prediction')
    ax.set_xlabel("Hour")
    ax.set_ylabel("Temperature [°C]")
    ax.set_title(f"Temperature forecast vs actual, batch {batch_num}")
    ax.legend()
    fig.tight_layout()
    fig.show()
    if save_fig:
        fig.savefig(os.path.join(folderpath, f"{filename}"))


def voltage_image_plot(deviations, max_deviation, figure_path, season, save_suffix):
    # Create subplots for three images side by side
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))

    for p in range(3):
        # Plot the image
        ax = axes[p]
        im = ax.imshow(deviations[p].T, cmap='RdBu_r', aspect='auto', vmin=-max_deviation, vmax=max_deviation)
        ax.set_title(f"Voltage Deviation - Phase {'abc'[p]} [p.u.]")
        # Customize the number of ticks and labels
        step = 6  # one tick every 6h
        xticks = np.arange(0, deviations[p].T.shape[1] + 1,
                           (deviations[p].T.shape[1] / 24) * step)  # 5 ticks (one every 6h)
        ax.set_xticks(xticks)
        ax.set_xticklabels(np.arange(0, 25, step))

        if p == 1:
            ax.set_xlabel('Time [h]')
        if p == 0:
            ax.set_ylabel('Buses')

    # Create a common color bar for all subplots
    cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cax)

    plt.savefig(join(figure_path, normpath('Voltages' + season + save_suffix)), bbox_inches='tight')
    plt.show()



