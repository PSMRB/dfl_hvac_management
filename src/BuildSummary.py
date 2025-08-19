"""
For all result folders in a given folder, read the summary.pkl files and gather them in a single dataframe.
"""

import os
import pandas as pd
import pickle

def build_summary():
    paths = {}

    # cvxpylayer_macpath = os.path.normpath('/Users/pietro/Library/CloudStorage/OneDrive-JohnsHopkins/DFL2_HP_CECI/SmallOffice/Models/cvxpylayer')
    paths["cvxpylayer"] = os.path.abspath("data/SmallOffice/Models/cvxpylayer")
    paths["RC"] = os.path.join(paths["cvxpylayer"], "RC")
    paths["spatialRC"] = os.path.join(paths["cvxpylayer"], "spatialRC")
    paths["NN1"] = os.path.join(paths["cvxpylayer"], "NotSparse/NN_1layers_2neurons")
    paths["NN2"] = os.path.join(paths["cvxpylayer"], "NotSparse/NN_1layers_5neurons")
    paths["NN3"] = os.path.join(paths["cvxpylayer"], "NotSparse/NN_1layers_10neurons")
    # paths["NN4"] = os.path.join(paths["cvxpylayer"], "NotSparse/NN_3layers_10neurons")
    # paths["NN5"] = os.path.join(paths["cvxpylayer"], "NotSparse/NN_1layers_10neurons")
    # paths["NN6"] = os.path.join(paths["cvxpylayer"], "NotSparse/NN_2layers_5neurons")

    # remove all the non-existing paths
    for key in list(paths.keys()):
        if not os.path.exists(paths[key]):
            del paths[key]
            print(f"Path {key} does not exist and has been removed from the paths dictionary.")

    # List all subfolders in P
    subfolders = [os.path.join(P, name) for P in paths.values() for name in os.listdir(P)
                  if os.path.isdir(os.path.join(P, name))]

    summaries = []
    for folder in subfolders:
        summary_path = os.path.join(folder, 'summary.pkl')
        if os.path.isfile(summary_path):
            with open(summary_path, 'rb') as f:
                summary = pickle.load(f)
                # If the summary is a dict, convert to DataFrame row
                if isinstance(summary, dict):
                    summaries.append(pd.DataFrame([summary]))
                # If it's already a DataFrame, just append
                elif isinstance(summary, pd.DataFrame):
                    summaries.append(summary)
                else:
                    print(f"Unknown summary format in {summary_path}")

    # Concatenate all summaries into a single DataFrame and reorder columns
    if summaries:
        df = pd.concat(summaries, ignore_index=True)
        # sort the dictionary
        order = ['Date', 'Clusters', 'Bin. Formulation', 'Model carac.', 'Seed', 'SNR', 'Warm-start', 'Learning rate',
                 'gamma', 'Update Frequency', 'Std w', 'Std b', 'N_ts', 'Nb Epoch', 'Best Epoch', 'Loss metric',
                 'Training time', 'Validation time', 'Test time', 'Test MIP gap', 'Train Infsbl', 'Val Infsbl',
                 'Test Infsbl',
                 'Best Train Loss', 'Best Val Loss', 'Test Loss', 'Test Expost+', 'Test MAE', 'Test MSE',
                 'Test Expected Cost',
                 'Test Expost Cost', 'Test Cost Misestimation', 'Test Tin Penalty']
        df = df[order]
    else:
        print("No summary.pkl files found in subfolders.")

    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True, ascending=False)
    df.reset_index(inplace=True)
    print(df)
    df.to_excel(os.path.join(paths["cvxpylayer"], 'summary.xlsx'), index=True, header=True, na_rep="NaN")

if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../"))  # change to the root directory of the project
    build_summary()
    print("Summary built successfully.")

