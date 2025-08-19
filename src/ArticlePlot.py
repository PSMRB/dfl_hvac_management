"""
Script to plot the article's figures which are not already appearing as such in other scripts.
"""
import os.path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import library.Common as c
import numpy as np
import pandas as pd
import pickle
import scienceplots

# Set the default plt.show to 100 dpi but savefig to 600 to fit IEEE standards
plt.style.use(['science', 'ieee', 'grid', 'no-latex'])
plt.rcParams.update({'figure.dpi': '100', 'savefig.dpi': '600', "savefig.format": 'pdf',
                     'axes.xmargin': 0.05, 'axes.ymargin': 0.05})


####################
# Time-of-use plot
####################
ts_ems = pd.date_range(start="2021-01-01", end="2021-01-02", freq="H")
prices = np.array([*[0.3] * 6, *[0.6] * 13, *[0.3] * 6])
fig, ax = plt.subplots(1, 1, figsize=(3.3, 2))
myFmt = mdates.DateFormatter('%H:%M')  # here you can format your datetick labels as desired
plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=6))
plt.gca().xaxis.set_major_formatter(myFmt)
ax.step(ts_ems, prices, where="post")
# ax.plot(ts_ems, prices, label='Feed-in price')
ax.set_xlim(left=ts_ems[0], right=ts_ems[-1])
ax.set_ylabel('ToU Price [$/kWh]')
ax.set_xlabel('Time')
fig.tight_layout()
# fig.subplots_adjust(bottom=0.15)
fig.savefig(os.path.normpath("../Article/ToUtariff.pdf"))
fig.show()

####################
# Medoid cycle
####################

# Read the medoids days and reorder them to create a smooth transition between the days
folderpath = "optimization_parameters/1991-2020"
medoid_labels = pd.read_csv(os.path.join(folderpath, "Labels10Days.csv"), index_col=0,
                                    parse_dates=True)
medoids = pd.read_csv(os.path.join(folderpath, "Clusters10Days.csv"), index_col=0, parse_dates=True)
medoids = medoids.sort_values(by="Mean Tamb", ascending=True)
m_idx = medoids.index
cycle = [m_idx[i] for i in range(len(m_idx)) if i % 2 == 0] + [m_idx[i] for i in range(len(m_idx), -1, -1) if i % 2 == 1]
medoids = medoids.loc[cycle]

fig, ax = plt.subplots(1, 1)
xtick_labels = medoids.index
ax.bar(xtick_labels, medoids["Mean Tamb"], color="tab:blue", yerr=medoids["Std Tamb"], ecolor="black")
ax.set_xticklabels(xtick_labels, rotation=45, ha='right')
ax.set_ylabel("Ambient Temperature [°C]")
# fig.suptitle("Clusters")
ax.grid(False)
fig.tight_layout()
fig.savefig(os.path.normpath("../Figures/BarPlotClustersTemperatureOnly"))
fig.show()


####################
# Read the data
####################
# set the paths
historical = True
foldername = "2024-06-27_19h27m27s" if historical else "2024-06-26_20h47m22s"
folderpath = os.path.normpath(f"../BuildingModel/Models/cvxpylayer/RC/{foldername}")
power_cost_path = os.path.join(folderpath, "power_cost.pkl")
training_loss_path = os.path.join(folderpath, "training_losses.pkl")
validation_loss_path = os.path.join(folderpath, "validation_losses.pkl")

# load the files
with open(power_cost_path, "rb") as f:
    power_costs = pickle.load(f)
with open(training_loss_path, "rb") as f:
    training_losses = pickle.load(f)
with open(validation_loss_path, "rb") as f:
    validation_losses = pickle.load(f)

# if historical run, we want to retrieve the first recorded performance (corresponding to the warm-start model)
# otherwise, we want to retrieve the last recorded performance (corresponding to the final model)
index_to_retrieve = 0 if historical else -1
# retrieve last recorded performances
get = lambda x: x[index_to_retrieve]
last_power_costs = power_costs.applymap(get)  # convert to kW
last_training_losses = c.agg_itr_of_dict(training_losses[index_to_retrieve])
last_validation_losses = c.agg_itr_of_dict(validation_losses[index_to_retrieve])

# compute the mean of the performances
mean_power_costs = last_power_costs.applymap(np.mean)
mean_training_losses = c.map_dict(last_training_losses, np.mean)
mean_validation_losses = c.map_dict(last_validation_losses, np.mean)

debug = 1



