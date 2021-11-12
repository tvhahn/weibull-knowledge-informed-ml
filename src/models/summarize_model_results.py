import numpy as np
import pandas as pd
import seaborn as sns
import os
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from multiprocessing import Pool

import torch
from src.data.data_utils import load_train_test_ims, load_train_test_femto
from src.models.utils import (
    test_metrics_to_results_df,
)
from src.models.loss import RMSELoss, RMSLELoss

import h5py
from pathlib import Path

import os
from shutil import copyfile
import sys

from scipy.stats import pointbiserialr
import argparse


"""
Gather all the result csv's, combine them together, and then append the test scores.

Filter out poorly performing models and save the final result csv's in the models/final folder.

Save the top performing models also in the models/final folder.
"""

#######################################################
# Set Arguments
#######################################################

# parse arguments
parser = argparse.ArgumentParser()

parser.add_argument(
    "-s",
    "--data_set",
    dest="data_set",
    type=str,
    default="ims",
    help="The data set use (either 'ims' or 'femto')",
)

# parser.add_argument(
#     "-d", 
#     "--path_data", 
#     dest="path_data", 
#     type=str, 
#     help="Path to processed data"
# )



# parser.add_argument(
#     "-p",
#     "--proj_dir",
#     dest="proj_dir",
#     type=str,
#     help="Location of project folder",
# )

# parser.add_argument(
#     "--random_search_iter",
#     dest="random_search_iter",
#     type=int,
#     default=3000,
#     help="Number of random searches to iterate over",
# )

# parser.add_argument(
#     "--epochs",
#     dest="epochs",
#     type=int,
#     default=2000,
#     help="Number of epochs to train each model",
# )

# parser.add_argument(
#     "--patience",
#     dest="patience",
#     type=int,
#     default=50,
#     help="Number of epochs without change before quiting training",
# )

args = parser.parse_args()

# General Parameters
SAVE_ENTIRE_CSV = True  # if you want to save the entire CSV, before filtering
ADD_TEST_RESULTS = True  # if you want to append the test results
TOP_MODEL_COUNT = 2  # the number of models to save in models/final/top_models directory
# e.g. save top 10 models

# Filter parameters
R2_BOUND = 0.2  # greater than
RMSE_BOUND = 0.35  # less than
SORT_BY = "r2_test"  # metric used to evaluate results
# options include: 'loss_rmse_test', 'r2_val'
# 'r2_test_avg', etc.

DATASET_TYPE = args.data_set  # 'ims' or 'femto'

#####

# use multi-processing to load all the CSVs into one file
# https://stackoverflow.com/a/36590187
# wrap your csv importer in a function that can be mapped
def read_csv(filename):
    "converts a filename to a pandas dataframe"
    return pd.read_csv(filename)


def set_directories():
    """Sets the directory paths used for data, checkpoints, etc."""

    # check if "scratch" path exists in the home directory
    # if it does, assume we are on HPC
    scratch_path = Path.home() / "scratch"

    # set the default directories
    if scratch_path.exists():
        # set important folder locations
        print("Assume on HPC")
        root_dir = Path.cwd()
        print("#### Root dir:", root_dir)

        if DATASET_TYPE == "ims":
            folder_data = Path.cwd() / "data/processed/IMS/"
        else:
            folder_data = Path.cwd() / "data/processed/FEMTO/"

        folder_results = Path(
            scratch_path / f"weibull_results/results_csv_{DATASET_TYPE}"
        )

        folder_checkpoints = Path(
            scratch_path / f"weibull_results/checkpoints_{DATASET_TYPE}"
        )
        
        folder_learning_curves = Path(
            scratch_path / f"weibull_results/learning_curves_{DATASET_TYPE}"
        )

    else:
        # set important folder locations
        print("Assume on local compute")
        root_dir = Path.cwd()
        print("#### Root dir:", root_dir)

        # data folder
        if DATASET_TYPE == "ims":
            folder_data = root_dir / "data/processed/IMS/"
            print("load IMS data", folder_data)
        else:
            folder_data = root_dir / "data/processed/FEMTO/"
            print("load FEMTO data", folder_data)

        folder_results = root_dir / f"models/interim/results_csv_{DATASET_TYPE}"
        folder_checkpoints = root_dir / f"models/interim/checkpoints_{DATASET_TYPE}"
        folder_learning_curves = (
            root_dir / f"models/interim/learning_curves_{DATASET_TYPE}"
        )

    return (
        folder_results,
        folder_checkpoints,
        folder_learning_curves,
        folder_data,
        root_dir,
    )


def main(folder_results):

    # get a list of file names
    files = os.listdir(folder_results)
    file_list = [
        folder_results / filename for filename in files if filename.endswith(".csv")
    ]

    # set up your pool
    with Pool(processes=7) as pool:  # or whatever your hardware can support

        # have your pool map the file names to dataframes
        df_list = pool.map(read_csv, file_list)

        # reduce the list of dataframes to a single dataframe
        combined_df = pd.concat(df_list, ignore_index=True)

        return combined_df


if __name__ == "__main__":

    (
        folder_results,
        folder_checkpoints,
        folder_learning_curves,
        folder_data,
        root_dir,
    ) = set_directories()

    df = main(folder_results)

    # drop first column
    try:
        df = df.drop(columns="Unnamed: 0")
    except:
        pass

    # add a unique identifier for each model architecture
    df["date_time_seed"] = (
        df["date_time"].astype(str) + "_" + df["rnd_seed_input"].astype(str)
    )

    # get name that model checkpoint was saved under
    df["model_checkpoint_name"] = (
        df["date_time"].astype(str)
        + "_"
        + df["loss_func"]
        + "_"
        + df["rnd_seed_input"].astype(str)
        + ".pt"
    )

    if SAVE_ENTIRE_CSV:
        df.to_csv(
            root_dir / "models/final" / f"{DATASET_TYPE}_results_summary_all.csv.gz",
            index=False, compression="gzip",
        )

    #### append test results to df ####
    if ADD_TEST_RESULTS:

        if DATASET_TYPE == "ims":
            (
                x_train,
                y_train,
                x_val,
                y_val,
                x_test,
                y_test,
                x_train_2,
                y_train_2,
                x_train_3,
                y_train_3,
            ) = load_train_test_ims(folder_data)

        else:
            (
                x_train,
                y_train,
                x_val,
                y_val,
                x_test,
                y_test,
                x_train1_1,
                y_train1_1,
                x_train2_1,
                y_train2_1,
                x_train3_1,
                y_train3_1,
                x_val1_2,
                y_val1_2,
                x_val2_2,
                y_val2_2,
                x_val3_2,
                y_val3_2,
                x_test1_3,
                y_test1_3,
                x_test2_3,
                y_test2_3,
                x_test3_3,
                y_test3_3,
            ) = load_train_test_femto(folder_data)

        # load beta, eta for Weibull CDF
        with h5py.File(folder_data / "eta_beta_r.hdf5", "r") as f:
            eta_beta_r = f["eta_beta_r"][:]

        ETA = eta_beta_r[0]
        BETA = eta_beta_r[1]

        y_train_days = torch.reshape(y_train[:, 0], (-1, 1))
        y_val_days = torch.reshape(y_val[:, 0], (-1, 1))
        y_test_days = torch.reshape(y_test[:, 0], (-1, 1))

        y_train = torch.reshape(y_train[:, 1], (-1, 1))
        y_val = torch.reshape(y_val[:, 1], (-1, 1))
        y_test = torch.reshape(y_test[:, 1], (-1, 1))

        if DATASET_TYPE == "ims":
            y_train_days_2 = torch.reshape(y_train_2[:, 0], (-1, 1))
            y_train_days_3 = torch.reshape(y_train_3[:, 0], (-1, 1))
            y_train_2 = torch.reshape(y_train_2[:, 1], (-1, 1))
            y_train_3 = torch.reshape(y_train_3[:, 1], (-1, 1))

        # append test results onto results dataframe
        df = test_metrics_to_results_df(folder_checkpoints, df, x_test, y_test)

        standard_losses = ["mse", "rmse", "rmsle"]

        # apply 0 or 1 for weibull, and for each unique loss func
        for index, value in df["loss_func"].items():
            if value in standard_losses:
                df.loc[index, "weibull_loss"] = 0
            else:
                df.loc[index, "weibull_loss"] = 1

        # convert to 'weibull_loss' column to integer
        df["weibull_loss"] = df["weibull_loss"].astype(int)

        # 0 of no dropping is used, otherwise 1
        for index, value in df["prob_drop"].items():
            if value > 0:
                df.loc[index, "prob_drop_true"] = 1
            else:
                df.loc[index, "prob_drop_true"] = 0

        df["prob_drop_true"] = df["prob_drop_true"].astype(int)

        loss_func_list = df["loss_func"].unique()

        for index, value in df["loss_func"].items():
            for loss_func in loss_func_list:
                df.loc[index, value] = 1

        df[loss_func_list] = df[loss_func_list].fillna(0, downcast="infer")

        if SAVE_ENTIRE_CSV:
            df.to_csv(
                root_dir / "models/final" / f"{DATASET_TYPE}_results_summary_all.csv.gz",
                index=False, compression="gzip",
            )
    # how many unique model architectures?
    print("No. unique model architectures:", len(df["date_time_seed"].unique()))
    print(
        "No. unique models (includes unique loss functions):", len(df["date_time_seed"])
    )

    ##### Filter resutls and select top models #####
    loss_func_list = df["loss_func"].unique()

    sort_by = SORT_BY

    dfr = df[
        (df["r2_test"] > R2_BOUND)
        & (df["loss_rmse_test"] < RMSE_BOUND)
        & (df["r2_train"] > R2_BOUND)
        & (df["loss_rmse_train"] < RMSE_BOUND)
        & (df["r2_val"] > R2_BOUND)
        & (df["loss_rmse_val"] < RMSE_BOUND)
        & (df["beta"] == 2.0)
    ][:]

    dfr = (
        dfr.groupby(["date_time_seed"])
        .apply(lambda x: x.sort_values([sort_by], ascending=False))
        .reset_index(drop=True)
    )
    dfr = (
        dfr.groupby(["date_time_seed"]).head(1).sort_values(by=sort_by, ascending=False)
    )

    # save filtered results csv
    dfr.to_csv(
        root_dir / "models/final" / f"{DATASET_TYPE}_results_filtered.csv", index=False
    )

    # create and save early stopping summary statistics
    df0 = dfr[dfr["weibull_loss"] == 0][["epoch_stopped_on"]].describe()
    df0 = df0.append(
        pd.DataFrame(
            [dfr[dfr["weibull_loss"] == 0][["epoch_stopped_on"]].median()],
            index=["median"],
        )
    )
    df0.columns = ["trad_loss_func"]

    df1 = dfr[dfr["weibull_loss"] == 1][["epoch_stopped_on"]].describe()
    df1 = df1.append(
        pd.DataFrame(
            [dfr[dfr["weibull_loss"] == 1][["epoch_stopped_on"]].median()],
            index=["median"],
        )
    )
    df1.columns = ["weibull_loss_func"]

    df_summary = df0.merge(df1, left_index=True, right_index=True)
    df_summary.to_csv(
        root_dir / "models/final" / f"{DATASET_TYPE}_early_stop_summary_stats.csv",
        index=True,
    )

    # select top N models and save in models/final/top_models directory
    top_models = dfr["model_checkpoint_name"][:TOP_MODEL_COUNT]
    Path(root_dir / f"models/final/top_models_{DATASET_TYPE}").mkdir(
        parents=True, exist_ok=True
    )
    top_model_folder = root_dir / f"models/final/top_models_{DATASET_TYPE}"
    for model_name in top_models:
        copyfile(
            folder_checkpoints / f"{model_name}", top_model_folder / f"{model_name}"
        )
        learning_curve = model_name.split(".")[0]
        copyfile(
            folder_learning_curves / f"{learning_curve}.png",
            top_model_folder / f"{learning_curve}.png",
        )

    # copy model.py in src/models/ to the models/final/top_models directory so that we can
    # easily load the saved checkpoints for later
    copyfile(root_dir / "src/models/model.py", top_model_folder / "model.py")

    # count up how often each loss functions type appears as a top performer
    def change_loss_func_name(cols):
        loss_func = cols[0]

        if loss_func == "mse":
            return "MSE"
        elif loss_func == "rmse":
            return "RMSE"
        elif loss_func == "rmsle":
            return "RMSLE"
        elif loss_func == "weibull_mse":
            return "Weibull-MSE\nCombined"
        elif loss_func == "weibull_rmse":
            return "Weibull-RMSE\nCombined"
        elif loss_func == "weibull_rmsle":
            return "Weibull-RMSLE\nCombined"
        elif loss_func == "weibull_only_mse":
            return "Weibull Only MSE"
        elif loss_func == "weibull_only_rmse":
            return "Weibull Only RMSE"
        else:
            return "Weibull Only RMLSE"

    df_count = (
        dfr.groupby(["loss_func"], as_index=False)
        .count()[["loss_func", "date_time"]]
        .rename(columns={"date_time": "count"})
        .sort_values(by="count", ascending=False)
    )
    df_count["loss_func2"] = df_count[["loss_func"]].apply(
        change_loss_func_name, axis=1
    )
    df_count = df_count.drop("loss_func", axis=1)
    df_count = df_count.rename(columns={"loss_func2": "loss_func"})
    df_count["count"] = df_count["count"].astype(float)
    df_count["percent"] = 100 * df_count["count"] / df_count["count"].sum()

    # save csv so we can use it later to create charts with
    df_count.to_csv(
        root_dir / "models/final" / f"{DATASET_TYPE}_count_results.csv", index=False
    )

    # perform correlation analysis over the various loss functions
    dfr = df[
        (df["r2_test"] > R2_BOUND)
        & (df["loss_rmse_test"] < RMSE_BOUND)
        & (df["r2_train"] > R2_BOUND)
        & (df["loss_rmse_train"] < RMSE_BOUND)
        & (df["r2_val"] > R2_BOUND)
        & (df["loss_rmse_val"] < RMSE_BOUND)
        & (df["beta"] == 2.0)
    ][:]

    def change_loss_func_name_corr(cols):
        loss_func = cols[0]

        if loss_func == "mse":
            return "MSE"
        elif loss_func == "rmse":
            return "RMSE"
        elif loss_func == "rmsle":
            return "RMSLE"
        elif loss_func == "weibull_mse":
            return "Weibull-MSE\nCombined"
        elif loss_func == "weibull_rmse":
            return "Weibull-RMSE\nCombined"
        elif loss_func == "weibull_rmsle":
            return "Weibull-RMSLE\nCombined"
        elif loss_func == "weibull_only_mse":
            return "Weibull Only\nMSE"
        elif loss_func == "weibull_only_rmse":
            return "Weibull Only\nRMSE"
        else:
            return "Weibull Only\nRMLSE"

    df_c = dfr[list(loss_func_list) + [sort_by]].copy()

    results = {}
    for i in loss_func_list:
        results[i] = list(pointbiserialr(df_c[i], df_c[sort_by]))

    df_corr = pd.DataFrame.from_dict(results).T
    df_corr = (
        df_corr.rename(columns={0: "corr", 1: "p_value"})
        .sort_values(by="corr")
        .sort_values(by="corr", ascending=False)
    )
    df_corr["loss_func2"] = df_corr.index  # reset index
    df_corr = df_corr.reset_index(drop=True)
    df_corr["loss_func"] = df_corr[["loss_func2"]].apply(
        change_loss_func_name_corr, axis=1
    )
    df_corr = df_corr.drop("loss_func2", axis=1)
    df_corr.to_csv(
        root_dir / "models/final" / f"{DATASET_TYPE}_correlation_results.csv",
        index=False,
    )
