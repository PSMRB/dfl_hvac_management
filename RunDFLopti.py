"""
Run the DFLopti optimization in parallel on any computer.
"""

import sys
from time import sleep
from src.GenerateDatasets import main_generate_datasets
from src.IndoorTemperatureModelling import main_indoor_temperature_modelling
from src.Clustering import main_clustering
from src.DFLoptiTaskAndParams import create_parameter_list, manual_modification_of_parameters, \
    filter_params_wrapper, split_params, parallel_dfl_training
from src.BuildSummary import build_summary


# 1) Generate the datasets of the 6 zones small office model in Denver, CO, USA.
main_generate_datasets()
# 2) Train some machine learning models to predict the indoor temperature on historical data 1991-2020.
main_indoor_temperature_modelling()
# 3) Perform the clustering of the data to find the medoids.
main_clustering()

# 4) Create the parameter sets to run the DFLopti optimization.
# create the list of parameter sets to run
params_all = create_parameter_list()
# Modify the parameters manually if needed to improve the performance of the training
params_all = manual_modification_of_parameters(params_all)
# Filter the parameters based on the parameters already run (successfully in summary.xlsx or failed in error_log)
filtered_params = filter_params_wrapper(params_all, "data/SmallOffice/Models/cvxpylayer/summary.xlsx")
# Split the parameters into chunks for parallel processing by the array job and save them to a file
split_params(filtered_params, 1)

# 5) Run the DFLopti optimization in parallel
parallel_dfl_training(0)
# 6) Build the summary of the results
build_summary()
