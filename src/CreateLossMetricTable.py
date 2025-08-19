"""
Script to read the loss metrics of the training, validation and test sets and create a table with the test results.
The table contains the test metrics for the best training of each model (NN1, NN2, RC) and binary formulation
(QP relaxation, Fixed binary, Stochastic Smoothing).
"""
import os.path
from src.library.CreateLossMetricTable_Library import *

def create_loss_metric_table():
    ### Parameters
    # os.chdir("/Users/pietro/Library/CloudStorage/OneDrive-JohnsHopkins/DFL2_HP_CECI/Script")
    savepath = os.path.normpath("output/TestLossTable")  # where to save the table
    if not os.path.exists(os.path.dirname(savepath)):
        os.mkdir(os.path.dirname(savepath))  # create the folder if it does not exist
    summaryresults_folderpath = os.path.normpath("data/SmallOffice/Models/cvxpylayer")  # where to find the excel with the result summary
    allresults_folderpath = os.path.normpath("data/SmallOffice/Models")
    summaryresults_filepath = os.path.join(summaryresults_folderpath, "summary.xlsx")  # the excel file with the summary of the results
    paths = {"savepath": savepath, "summaryresults_folderpath": summaryresults_folderpath,
             "allresults_folderpath": allresults_folderpath, "summaryresults_filepath": summaryresults_filepath}

    # create the table
    cols = pd.MultiIndex.from_arrays([[*["RC"]*2, *["NN1"]*3, *["NN2"]*3, "NN3"], ["SS", "QP", "SS", "QP", "FB", "SS", "QP", "FB", "QP"]], names=("Model", "Formulation"))
    # idx = ["Hierarchical loss", "MAE (kW)", "MSE (kW^2)", "Error mean (kW)", "Error std (kW)",
    #                "Expected cost ($)", "Ex-post cost ($)", "Cost error ($)", "Nb. Epochs", "Training time"]
    idx = ["Ex-post+ ($)", "Hierarchical loss",
           "MAE (kW)", "MSE (kW2)", "Error mean (kW)", "Error std (kW)",
             "Expected cost ($)", "Ex-post cost ($)", "Cost error ($)", "Temp. Penalty($)",
           "Nb. Epochs", "Training time", "Validation time", "Test time"]
    table1 = pd.DataFrame(columns=cols, index=idx)

    ### extract the best trainings
    loaded_df = pd.read_excel(summaryresults_filepath, index_col=0, header=0)
    feasible_df = loaded_df[(loaded_df["Val Infsbl"] == 0) & (loaded_df["Test Infsbl"] == 0)]
    gb_model = feasible_df.groupby("Model carac.")  # group by the model characteristics (rcmodel, nn architecture, etc.)
    interesting_df = gb_model.filter(lambda x: x.name in ["rcmodel", "spatialrcmodel", "1layers_2neurons_each", "1layers_5neurons_each", "1layers_10neurons_each"])  # keep only the useful architectures: "spatialrcmodel",
    gb_modelandformulation = interesting_df.groupby("Model carac.").apply(lambda x: x.groupby("Bin. Formulation"))  # each model group is grouped by binary formulation

    # get the best training for each model and formulation, and fill the table
    table1 = get_best_and_fill_table(gb_modelandformulation, interesting_df, table1, paths, get_header1)

    ### TABLE 2
    paths["savepath"] = os.path.join(os.path.dirname(paths["savepath"]), "ITOvsDFL")  # where to save the table

    cols2 = pd.MultiIndex.from_arrays([[*["RC"]*2, *["NN1"]*2, *["NN2"]*2, *["NN3"]*2], ["ITO", "QP", "ITO", "SS", "ITO", "SS", "ITO", "QP"]], names=("Model", "Formulation"))
    table2 = pd.DataFrame(columns=cols2, index=table1.index)

    # Fill columns with the best results from training
    table2[("NN1", "SS")] = table1[("NN1", "SS")]
    table2[("NN2", "SS")] = table1[("NN2", "SS")]
    table2[("NN3", "QP")] = table1[("NN3", "QP")]
    table2[("RC", "QP")] = table1[("RC", "QP")]

    # to correct the +1 added to the epoch
    table2.loc["Nb. Epochs"] = table2.loc["Nb. Epochs"] - 1

    # Find corresponding ITO results
    interesting_df = interesting_df.loc[interesting_df["Nb Epoch"] == "0/1"]
    gb_modelandformulation = interesting_df.groupby("Model carac.").apply(lambda x: x.groupby("Model carac."))  # each model group is grouped by model

    # get the best training for each model and formulation, and fill the table
    table2 = get_best_and_fill_table(gb_modelandformulation, interesting_df, table2, paths, get_header2)


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../"))  # change to the root directory of the project
    create_loss_metric_table()
    print("Loss metric tables created successfully.")