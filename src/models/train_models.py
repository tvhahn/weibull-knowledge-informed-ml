import numpy as np
from pathlib import Path
import pandas as pd
import datetime
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.model_selection import ParameterSampler
from scipy.stats import randint as sp_randint
from scipy.stats import uniform

# import custom functions and classes
from utils import EarlyStopping
from src.data.data_utils import load_train_test_ims, load_train_test_femto
from model import Net
from loss import RMSELoss, RMSLELoss, WeibullLossRMSE, WeibullLossRMSLE, WeibullLossMSE
import h5py
from src.visualization.visualize_training import (
    plot_trained_model_results_ims,
    plot_trained_model_results_femto,
)
import argparse


#######################################################
# Argparse
#######################################################

# parse arguments
parser = argparse.ArgumentParser()

parser.add_argument(
    "-d", 
    "--path_data", 
    dest="path_data", 
    type=str, 
    help="Path to processed data"
)

parser.add_argument(
    "-s",
    "--data_set",
    dest="data_set",
    type=str,
    default="ims",
    help="The data set use (either 'ims' or 'femto')",
)

parser.add_argument(
    "-p",
    "--proj_dir",
    dest="proj_dir",
    type=str,
    help="Location of project folder",
)

parser.add_argument(
    "--random_search_iter",
    dest="random_search_iter",
    type=int,
    default=3000,
    help="Number of random searches to iterate over",
)

parser.add_argument(
    "--epochs",
    dest="epochs",
    type=int,
    default=2000,
    help="Number of epochs to train each model",
)

parser.add_argument(
    "--patience",
    dest="patience",
    type=int,
    default=50,
    help="Number of epochs without change before quiting training",
)

args = parser.parse_args()

###################
# Set Constants
###################

# before random search
RANDOM_SEARCH_ITERATIONS = args.random_search_iter
EPOCHS = args.epochs
PATIENCE = args.patience
EARLY_STOP_DELAY = 0


#######################################################
# Set Directories
#######################################################

# set project directory
if args.proj_dir:
    proj_dir = Path(args.proj_dir)
else:
    # proj_dir assumed to be cwd
    proj_dir = Path.cwd()


# check if "scratch" path exists in the home directory
# if it does, assume we are on HPC
scratch_path = Path.home() / "scratch"
if scratch_path.exists():
    print("Assume on HPC")
else:
    print("Assume on local compute")

# set random seed for parameter search
if scratch_path.exists():
    print('#### Running on HPC')
    # for HPC input
    DATASET_TYPE = args.data_set  # 'ims' or 'femto'
    RANDOM_SEED_INPUT = np.random.randint(0, 1e7)
    print("RANDOM_SEED_INPUT = ", RANDOM_SEED_INPUT)

    # set important folder locations
    Path(scratch_path / "weibull_results").mkdir(parents=True, exist_ok=True)
    Path(scratch_path / f"weibull_results/learning_curves_{DATASET_TYPE}").mkdir(
        parents=True, exist_ok=True
    )
    Path(scratch_path / f"weibull_results/results_csv_{DATASET_TYPE}").mkdir(
        parents=True, exist_ok=True
    )
    Path(scratch_path / f"weibull_results/checkpoints_{DATASET_TYPE}").mkdir(
        parents=True, exist_ok=True
    )
    folder_path = scratch_path / "weibull_results"

    print("#### FOLDER_PATH:", folder_path)

    if DATASET_TYPE == "ims":
        folder_data = Path(args.path_data) / "processed/IMS"
        print('Folder data path:', folder_data)
    else:
        folder_data = Path(args.path_data) / "processed/FEMTO"
        print('Folder data path:', folder_data)

    folder_results = Path(scratch_path / f"weibull_results/results_csv_{DATASET_TYPE}")
    folder_checkpoints = Path(
        scratch_path / f"weibull_results/checkpoints_{DATASET_TYPE}"
    )
    folder_learning_curves = Path(
        scratch_path / f"weibull_results/learning_curves_{DATASET_TYPE}"
    )

else:
    # if not on HPC then on local comp
    DATASET_TYPE = args.data_set   # 'ims' or 'femto'
    RANDOM_SEED_INPUT = np.random.randint(0, 1e7)
    # RANDOM_SEED_INPUT = 12

    # set important folder locations
    folder_path = proj_dir
    print("folder_path -->", folder_path)
    Path(folder_path / f"models/interim/learning_curves_{DATASET_TYPE}").mkdir(
        parents=True, exist_ok=True
    )
    Path(folder_path / f"models/interim/results_csv_{DATASET_TYPE}").mkdir(
        parents=True, exist_ok=True
    )
    Path(folder_path / f"models/interim/checkpoints_{DATASET_TYPE}").mkdir(
        parents=True, exist_ok=True
    )

    print("#### FOLDER_PATH:", folder_path)

    # data folder
    if DATASET_TYPE == "ims":
        folder_data = Path(args.path_data) / "IMS"
        print("load IMS data", folder_data)
    else:
        folder_data = Path(args.path_data) / "FEMTO"
        print("load FEMTO data", folder_data)

    folder_results = folder_path / f"models/interim/results_csv_{DATASET_TYPE}"
    folder_checkpoints = folder_path / f"models/interim/checkpoints_{DATASET_TYPE}"
    folder_learning_curves = (
        folder_path / f"models/interim/learning_curves_{DATASET_TYPE}"
    )

######################################
# Define Parameters for Random Search
######################################

# parameter grid to search over
param_grid = {
    "batch_size": [32, 64, 128, 256, 512],
    "learning_rate": [0.1, 0.01, 0.001, 0.0001],
    "lambda_mod": uniform(loc=0, scale=3),
    "n_layers": [2, 3, 4, 5, 6, 7],
    "n_units": [16, 32, 64, 128, 256],
    "prob_drop": [0.6, 0.5, 0.4, 0.25, 0.2, 0.1, 0],
    # "prob_drop": [0],
    # "beta": [1.3, 1.5, 1.7, 1.8, 2.0, 2.1, 2.3],
    "beta": [2.0],
}

# generate parameter list
param_list = list(
    ParameterSampler(
        param_grid,
        n_iter=RANDOM_SEARCH_ITERATIONS,
        random_state=np.random.RandomState(RANDOM_SEED_INPUT),
    )
)


# select device to run neural net on
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on GPU")
else:
    device = torch.device("cpu")
    print("Running on CPU")


###################
# Load Data Set
###################


# load the train/val/test sets. will be loaded as tensors
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

    y_train_days = torch.reshape(y_train[:, 0], (-1, 1))
    y_val_days = torch.reshape(y_val[:, 0], (-1, 1))

    # make sure % remainin life is selected for final
    # train/validation set used in training
    y_train = torch.reshape(y_train[:, 1], (-1, 1))
    y_val = torch.reshape(y_val[:, 1], (-1, 1))

    y_train_2 = torch.reshape(y_train_2[:, 1], (-1, 1))
    y_train_3 = torch.reshape(y_train_3[:, 1], (-1, 1))

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

    y_train_days = torch.reshape(y_train[:, 0], (-1, 1))
    y_val_days = torch.reshape(y_val[:, 0], (-1, 1))

    # make sure % remainin life is selected for final
    # train/validation set used in training
    y_train = torch.reshape(y_train[:, 1], (-1, 1))
    y_val = torch.reshape(y_val[:, 1], (-1, 1))

    y_train1_1 = torch.reshape(y_train1_1[:, 1], (-1, 1))
    y_train2_1 = torch.reshape(y_train2_1[:, 1], (-1, 1))
    y_train3_1 = torch.reshape(y_train3_1[:, 1], (-1, 1))
    y_val1_2 = torch.reshape(y_val1_2[:, 1], (-1, 1))
    y_val2_2 = torch.reshape(y_val2_2[:, 1], (-1, 1))
    y_val3_2 = torch.reshape(y_val3_2[:, 1], (-1, 1))


# load beta, eta for Weibull CDF
with h5py.File(folder_data / "eta_beta_r.hdf5", "r") as f:
    eta_beta_r = f["eta_beta_r"][:]

# load the t_array in case we want to try different beta's
with h5py.File(folder_data / "t_array.hdf5", "r") as f:
    t_array = f["t_array"][:]


ETA = eta_beta_r[0]
BETA = eta_beta_r[1]
R = eta_beta_r[2]
# print("BETA: ", BETA)
# print("ETA: ", ETA)
# print("R: ", R)


###################
# Functions
###################


def create_eta(t_array, beta, r=2):
    # characteristic life
    eta = (np.sum((t_array ** beta) / r)) ** (1 / beta)
    return eta


def fwd_pass(
    net,
    x,
    y,
    y_days,
    optimizer,
    train=False,
    loss_func="mse",
    lambda_mod=1.0,
    eta=13.0,
    beta=2.0,
):
    """Similar to Sentdex tutorial
    https://pythonprogramming.net/analysis-visualization-deep-learning-neural-network-pytorch/
    """
    if train:
        net.zero_grad()

    y_hat = net(x)

    if loss_func == "rmse":
        criterion = RMSELoss()
        loss = criterion(y_hat, y)
    elif loss_func == "rmsle":
        criterion = RMSLELoss()
        loss = criterion(y_hat, y)
    elif loss_func == "weibull_rmse":
        criterion_rmse = RMSELoss()
        criterion_weibull = WeibullLossRMSE()
        loss = criterion_rmse(y_hat, y) + criterion_weibull(
            y_hat, y, y_days, lambda_mod=lambda_mod, eta=eta, beta=beta
        )
    elif loss_func == "weibull_rmsle":
        criterion_rmsle = RMSLELoss()
        criterion_weibull = WeibullLossRMSLE()
        loss = criterion_rmsle(y_hat, y) + criterion_weibull(
            y_hat, y, y_days, lambda_mod=lambda_mod, eta=eta, beta=beta
        )
    elif loss_func == "weibull_only_rmse":
        criterion_weibull = WeibullLossRMSE()
        loss = criterion_weibull(
            y_hat, y, y_days, lambda_mod=lambda_mod, eta=eta, beta=beta
        )
    elif loss_func == "weibull_only_rmsle":
        criterion_weibull = WeibullLossRMSLE()
        loss = criterion_weibull(
            y_hat, y, y_days, lambda_mod=lambda_mod, eta=eta, beta=beta
        )
    elif loss_func == "weibull_mse":
        criterion_mse = nn.MSELoss()
        criterion_weibull = WeibullLossMSE()
        loss = criterion_mse(y_hat, y) + criterion_weibull(
            y_hat, y, y_days, lambda_mod=lambda_mod, eta=eta, beta=beta
        )

    elif loss_func == "weibull_only_mse":
        criterion_weibull = WeibullLossMSE()
        loss = criterion_weibull(
            y_hat, y, y_days, lambda_mod=lambda_mod, eta=eta, beta=beta
        )
    else:
        criterion = nn.MSELoss()
        loss = criterion(y_hat, y)

    if train:
        loss.backward()
        optimizer.step()

    return loss


def train(
    net,
    x_train,
    y_train,
    y_train_days,
    x_val,
    y_val,
    y_val_days,
    optimizer,
    loss_func="mse",
    batch_size=100,
    epochs=500,
    patience=7,
    lambda_mod=1.0,
    eta=13.0,
    beta=2.0,
    early_stop_delay=20,
    checkpoint_path="checkpoint.pt",
):

    df = pd.DataFrame()

    if loss_func in [
        "rmse",
        "rmsle",
        "weibull_rmse",
        "weibull_rmsle",
    ]:
        # initialize the early_stopping object
        early_stopping = EarlyStopping(
            patience=patience,
            verbose=False,
            early_stop_delay=early_stop_delay,
            path=checkpoint_path,
            delta=0.0001,
        )
    else:
        early_stopping = EarlyStopping(
            patience=patience,
            verbose=False,
            early_stop_delay=early_stop_delay,
            path=checkpoint_path,
            delta=0.00001,
        )

    for epoch in range(epochs):

        # track the training/validation losses during epoch
        train_losses = []
        train_losses_mse = []

        #############
        # train model
        #############
        for i in range(0, len(x_train), batch_size):

            # create the batches and send to GPU (or CPU)
            # implement data loader in the future
            batch_x = x_train[i : i + batch_size].to(device)
            batch_y = y_train[i : i + batch_size].to(device)
            batch_y_days = y_train_days[i : i + batch_size].to(device)

            # train and calculate the losses
            net.train()
            loss = fwd_pass(
                net,
                batch_x,
                batch_y,
                batch_y_days,
                optimizer,
                train=True,
                loss_func=loss_func,
                lambda_mod=lambda_mod,
                eta=eta,
                beta=beta,
            )
            net.eval()
            loss_mse = fwd_pass(
                net,
                batch_x,
                batch_y,
                batch_y_days,
                optimizer,
                train=False,
                loss_func="mse",
                lambda_mod=lambda_mod,
                eta=eta,
                beta=beta,
            )
            train_losses.append(loss.item())
            train_losses_mse.append(loss_mse.item())

        ################
        # validate model
        ################
        net.eval()
        val_loss = fwd_pass(
            net,
            x_val.to(device),
            y_val.to(device),
            y_val_days.to(device),
            optimizer,
            train=False,
            loss_func=loss_func,
            lambda_mod=lambda_mod,
            eta=eta,
            beta=beta,
        )
        val_loss_mse = fwd_pass(
            net,
            x_val.to(device),
            y_val.to(device),
            y_val_days.to(device),
            optimizer,
            train=False,
            loss_func="mse",
            lambda_mod=lambda_mod,
            eta=eta,
            beta=beta,
        )

        loss_avg = np.mean(train_losses)
        loss_avg_mse = np.mean(train_losses_mse)

        # save the results to a pandas dataframe
        df = df.append(
            pd.DataFrame(
                [
                    [
                        epoch + 1,
                        loss_avg,
                        val_loss.item(),
                        loss_avg_mse,
                        val_loss_mse.item(),
                    ]
                ],
                columns=["epoch", "loss", "val_loss", "loss_mse", "val_loss_mse"],
            )
        )

        early_stopping(val_loss, net)

        if early_stopping.early_stop:
            print("Early stopping")
            break

        # print out the epoch, loss, and iteration number every 5th epoch
        if epoch % 200 == 0:
            print(f"Epoch: {epoch} \tLoss: {loss_avg:.4f} \tVal Loss: {val_loss:.4f}")

    # load the last checkpoint with the best model
    print("Load best model")
    net = torch.load(checkpoint_path)
    # net.load_state_dict(torch.load(checkpoint_path))

    df = df.reset_index(drop=True)

    return df, net


###################
# Training Loop
###################


# create dataframe to store all the results
col = [
    "date_time",
    "data_set",
    "loss_func",
    "rnd_seed_input",
    "rnd_search_iter",
    "rnd_search_iter_no",
    "beta",
    "eta",
    "epochs",
    "patience",
    "early_stop_delay",
    "batch_size",
    "learning_rate",
    "lambda_mod",
    "n_layers",
    "n_units",
    "prob_drop",
    "epoch_stopped_on",
]

# instantiate dataframe for storing results
df_results = pd.DataFrame()

# loop through each parameter
for i, param in enumerate(param_list):

    # set parameters
    BATCH_SIZE = param["batch_size"]
    LEARNING_RATE = param["learning_rate"]
    LAMBDA_MOD = param["lambda_mod"]
    N_LAYERS = param["n_layers"]
    N_UNITS = param["n_units"]
    PROB_DROP = param["prob_drop"]
    BETA = param["beta"]

    ETA = create_eta(t_array, BETA, R)

    # print("BETA CALCULATED: ", BETA)
    # print("ETA CALCULATED: ", ETA)

    # record time of model creation so we can uniquely identify
    # each random search iteration
    # format example: '2021_02_01_133931'
    date_time = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")

    # get date string for the final result csv
    if i == 0:
        date_results = date_time

    # iterate through each unique loss function
    # so that we can compare well between them all
    # loss functions are: 'rmse', 'mse', 'weibull'
    for LOSS_FUNCTION in [
        "mse",
        "rmse",
        "rmsle",
        "weibull_mse",
        "weibull_rmse",
        "weibull_rmsle",
        "weibull_only_mse",
        "weibull_only_rmse",
        "weibull_only_rmsle",
    ]:

        print(LOSS_FUNCTION)

        net = Net(x_train.shape[1], N_LAYERS, N_UNITS, PROB_DROP)

        net.to(device)
        optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

        # set the checkpoint name, and make it unique
        checkpoint_name = f"{date_time}_{LOSS_FUNCTION}_{RANDOM_SEED_INPUT}.pt"

        # save the results in a dataframe
        df, net = train(
            net,
            x_train,
            y_train,
            y_train_days,
            x_val,
            y_val,
            y_val_days,
            optimizer,
            loss_func=LOSS_FUNCTION,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            patience=PATIENCE,
            lambda_mod=LAMBDA_MOD,
            eta=ETA,
            beta=BETA,
            early_stop_delay=EARLY_STOP_DELAY,
            checkpoint_path=folder_checkpoints / checkpoint_name,
        )

        # plot the learning curves and save
        if DATASET_TYPE == "ims":
            (epoch_stopped_on, results_dict) = plot_trained_model_results_ims(
                df,
                net,
                x_train,
                y_train,
                x_train_2,
                y_train_2,
                x_train_3,
                y_train_3,
                x_val,
                y_val,
                device,
                date_time,
                folder_learning_curves,
                loss_func=LOSS_FUNCTION,
                batch_size=BATCH_SIZE,
                epochs=EPOCHS,
                patience=PATIENCE,
                lambda_mod=LAMBDA_MOD,
                learning_rate=LEARNING_RATE,
                eta=ETA,
                beta=BETA,
                n_layers=N_LAYERS,
                n_units=N_UNITS,
                prob_drop=PROB_DROP,
                early_stop_delay=EARLY_STOP_DELAY,
                save_pic=True,
                show_pic=False,
                rnd_seed=RANDOM_SEED_INPUT,
                data_set=DATASET_TYPE,
            )

            # create a list of the results that
            # will be appended onto the df_results dataframe
            results_list = [
                date_time,
                DATASET_TYPE,
                LOSS_FUNCTION,
                RANDOM_SEED_INPUT,
                RANDOM_SEARCH_ITERATIONS,
                i,
                BETA,
                ETA,
                EPOCHS,
                PATIENCE,
                EARLY_STOP_DELAY,
                BATCH_SIZE,
                LEARNING_RATE,
                LAMBDA_MOD,
                N_LAYERS,
                N_UNITS,
                PROB_DROP,
                epoch_stopped_on,
            ]
        else:

            (epoch_stopped_on, results_dict) = plot_trained_model_results_femto(
                df,
                net,
                x_train,
                y_train,
                x_val,
                y_val,
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
                device,
                date_time,
                folder_learning_curves,
                loss_func=LOSS_FUNCTION,
                batch_size=BATCH_SIZE,
                epochs=EPOCHS,
                patience=PATIENCE,
                lambda_mod=LAMBDA_MOD,
                learning_rate=LEARNING_RATE,
                eta=ETA,
                beta=BETA,
                n_layers=N_LAYERS,
                n_units=N_UNITS,
                prob_drop=PROB_DROP,
                early_stop_delay=EARLY_STOP_DELAY,
                save_pic=True,
                show_pic=False,
                rnd_seed=RANDOM_SEED_INPUT,
                data_set=DATASET_TYPE,
            )

            # create a list of the results that
            # will be appended onto the df_results dataframe
            results_list = [
                date_time,
                DATASET_TYPE,
                LOSS_FUNCTION,
                RANDOM_SEED_INPUT,
                RANDOM_SEARCH_ITERATIONS,
                i,
                BETA,
                ETA,
                EPOCHS,
                PATIENCE,
                EARLY_STOP_DELAY,
                BATCH_SIZE,
                LEARNING_RATE,
                LAMBDA_MOD,
                N_LAYERS,
                N_UNITS,
                PROB_DROP,
                epoch_stopped_on,
            ]

        col_update = col + list(results_dict.keys())
        results_list_update = results_list + [results_dict[i] for i in results_dict]
        df_temp = pd.DataFrame([results_list_update], columns=col_update)

        df_results = df_results.append(df_temp)

        # update csv of results
        df_results.to_csv(
            folder_results / f"results_{date_results}_{RANDOM_SEED_INPUT}.csv",
            index=False,
        )
