from src.DFLoptiTaskAndParams import (paper_parameter_sets1, paper_parameter_sets2, paper_parameter_sets3,
                                      paper_parameter_sets4, parallel_dfl_training)
from src.CreateTables import create_tables
from src.BuildSummary import build_summary
import sys
from time import sleep

try:
    n_tasks = int(sys.argv[1])
    id_task = int(sys.argv[2])
    parameter_sets = int(sys.argv[3])
    n_jobs = -1
except IndexError:  # not on the cluster
    print("Index Error caught, setting default values")
    n_tasks = 1
    id_task = 1
    parameter_sets = 1
    n_jobs = 1

if id_task == 0:
    ## Create the parameter sets to run the DFLopti optimization
    if parameter_sets == 1:
        # the ITO results for the paper
        print("In parameter sets 1")
        paper_parameter_sets1(n_tasks)
    elif parameter_sets == 2:
        # Run the trainings (RC, NN1, NN2, and NN3 with all methods QP, FB, SS)
        print("In parameter sets 2")
        paper_parameter_sets2(n_tasks)
    elif parameter_sets == 3:
        # Run the analysis on the number of samples S
        print("In parameter sets 3")
        paper_parameter_sets3(n_tasks)
    elif parameter_sets == 4:
        # the (not) tight formulation tests (they are aside because they require a previous training in sets1)
        print("In parameter sets 4")
        paper_parameter_sets4(n_tasks)
else:
    # wait for the file to be written
    sleep(5)
# run the DFL training
parallel_dfl_training(id_task, n_jobs)
# Build the summary of the results
build_summary()
# Once all the trainings are done, create the tables for the paper
if parameter_sets == 4:
    # Create the table for the paper
    create_tables()