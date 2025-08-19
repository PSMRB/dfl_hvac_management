from src.DFLoptiTaskAndParams import paper_parameter_sets, parallel_dfl_training
from src.CreateLossMetricTable import create_loss_metric_table
from src.BuildSummary import build_summary
import sys
from time import sleep

# Create the parameter sets to run the DFLopti optimization
paper_parameter_sets(1)
# run the DFL training
parallel_dfl_training(0)
# Build the summary of the results
build_summary()
# Create the table for the paper
create_loss_metric_table()