import numpy as np
import matplotlib

# run matplotlib without display
# https://stackoverflow.com/a/4706614/9214620
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.models.utils import test
import torch

from sklearn.metrics import r2_score
from src.models.loss import RMSELoss, RMSLELoss
import torch.nn as nn

"""
Functions to visually inspect the models after they have been saved.
"""

def plot_trained_model_results_ims(
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
    save_path,
    loss_func="mse",
    batch_size=99999,
    epochs=99999,
    patience=99999,
    lambda_mod=99999.0,
    learning_rate=99999.0,
    eta=99999.0,
    beta=99999.0,
    n_layers=99999,
    n_units=99999,
    prob_drop=0,
    early_stop_delay=99999,
    save_pic=False,
    show_pic=False,
    rnd_seed=99999,
    data_set='ims',
):
    fig, ax = plt.subplots(3, 2, figsize=(14, 14), constrained_layout=True,)
    net.eval()

    results_dict = {}

    # establish evalutation metrics
    criterion_mae = nn.L1Loss()
    criterion_rmse = RMSELoss()
    criterion_rmsle = RMSLELoss()

    #### SELECTED LOSS FUNCTION
    # in sample loss
    ax[0][0].plot(
        df["epoch"], df["loss"], label="Loss", linewidth=2, alpha=0.4, color="#d73027"
    )

    # out of sample (validation) loss
    ax[0][0].plot(
        df["epoch"], df["val_loss"], label="Val Loss", linewidth=2, color="#d73027"
    )
    try:
        epoch_stopped_on = len(df)-patience
    except:
        epoch_stopped_on = len(df)


    ax[0][0].axvline(
        epoch_stopped_on,
        linestyle="--",
        color="black",
        label="Early Stopping Checkpoint",
        alpha=0.6,
    )

    ax[0][0].legend()
    ax[0][0].set_xlabel("epoch")
    ax[0][0].set_ylabel(f"{loss_func} loss")
    ax[0][0].set_title(f"{loss_func} loss")

    #### MSE LOSS FUNCTION
    # in sample loss
    ax[0][1].plot(
        df["epoch"], df["loss"], label="Loss", linewidth=2, alpha=0.4, color="#d73027"
    )

    # out of sample (validation) loss
    ax[0][1].plot(
        df["epoch"], df["val_loss_mse"], label="Val Loss", linewidth=2, color="#d73027"
    )
    ax[0][1].axvline(
        epoch_stopped_on,
        linestyle="--",
        color="black",
        label="Early Stopping Checkpoint",
        alpha=0.6,
    )

    ax[0][1].set_xlabel("epoch")
    ax[0][1].set_ylabel("mse loss")
    ax[0][1].set_title("mse loss")

    try:
        mse_value_val = df["val_loss_mse"].iloc[epoch_stopped_on - 1]
        mse_value_train = df["loss_mse"].iloc[epoch_stopped_on - 1]
    except:
        mse_value_val = df["val_loss_mse"][0]
        mse_value_train = df["loss_mse"][0]

    # get x and y axis limits
    x_min, x_max = ax[0][1].get_xlim()
    y_min, y_max = ax[0][1].get_ylim()

    print_text = f"MSE = {mse_value_val:.3f}"

    ax[0][1].text(
        (x_max - x_min) * 0.95 + x_min,
        y_max - (y_max - y_min) * 0.1,
        print_text,
        fontsize="medium",
        fontweight="semibold",
        verticalalignment="center",
        horizontalalignment="right",
        bbox={"facecolor": "gray", "alpha": 0.2, "pad": 6},
    )

    #### RUL/PERCENT CURVE TRAIN
    def calc_r2_avg(y_hats, y_val, index_sorted, window_size):
        y_hats_rolling_avg = np.convolve(np.array(y_hats[index_sorted]).reshape(-1), np.ones(window_size), 'valid') / window_size
        try:
            r2_val_avg = r2_score(np.array(y_val)[index_sorted][window_size-1:], y_hats_rolling_avg)
        except:
            r2_val_avg = 99999
        return r2_val_avg, y_hats_rolling_avg


    y_hats = test(net, x_train, device, 100)
    index_sorted = np.array(np.argsort(y_train, 0).reshape(-1))

    ax[1][0].plot(np.array(y_train)[index_sorted], label="True Life Percentage", alpha=0.5)
    ax[1][0].scatter(
        np.arange(0, len(y_hats), 1),
        y_hats[index_sorted],
        label="Predicted Life Percentage",
        alpha=0.4,
        c="#fc8d59",
        s=1,
    )
    ax[1][0].set_ylabel("life percentage")
    ax[1][0].legend(loc='upper left')

    ax[1][0].set_title("train (all) results")


    # calculate losses
    loss_rmse_train = criterion_rmse(y_hats, y_train)
    loss_mae_train = criterion_mae(y_hats, y_train)
    loss_rmsle_train = criterion_rmsle(y_hats, y_train)
    try:
        r2_train = r2_score(y_train, y_hats)
    except:
        r2_train = 99999
    
    print_text = f"RMSE = {loss_rmse_train:.3f}\nRMSLE = {loss_rmsle_train:.3f}\nMAE = {loss_mae_train:.3f}\nR2 = {r2_train:.3f}"

    results_dict['loss_mse_train'] = mse_value_train
    results_dict['loss_rmse_train'] = loss_rmse_train.item()
    results_dict['loss_mae_train'] = loss_mae_train.item()
    results_dict['loss_rmsle_train'] = loss_rmsle_train.item()
    results_dict['r2_train'] = r2_train

    # get x and y axis limits
    x_min, x_max = ax[1][0].get_xlim()
    y_min, y_max = ax[1][0].get_ylim()

    ax[1][0].text(
        (x_max - x_min) * 0.95 + x_min,
        y_max - (y_max - y_min) * 0.87,
        print_text,
        fontsize="medium",
        fontweight="semibold",
        verticalalignment="center",
        horizontalalignment="right",
        bbox={"facecolor": "gray", "alpha": 0.2, "pad": 6},
    )

    # bottom text showing model parameters

    bottom_text = f"batch_size = {batch_size},   epochs = {epochs},   patience = {patience} \nlambda_mod = {lambda_mod:.3f},   eta = {eta:.3f},   beta = {beta:.3f}\nn_layers = {n_layers},  n_units = {n_units},   learning_rate = {learning_rate}\nprob_drop = {prob_drop},   early_stop_delay = {early_stop_delay},   epoch_stopped_on = {epoch_stopped_on} \n "

    ax[2][0].text(
        x_min + x_max * 0.05,
        y_min - y_max * 0.15,
        bottom_text,
        fontsize="medium",
        fontweight="normal",
        verticalalignment="top",
        horizontalalignment="left",
    )

    #### RUL/PERCENT CURVE VAL
    y_hats = test(net, x_val, device, 100)
    index_sorted = np.array(np.argsort(y_val, 0).reshape(-1))

    window_size = 12 # 2 hour rolling avg
    r2_val_avg, y_hats_rolling_avg = calc_r2_avg(y_hats, y_val, index_sorted, window_size)



    ax[1][1].plot(np.array(y_val)[index_sorted], alpha=0.5)
    ax[1][1].plot(np.arange(0, len(y_hats), 1)[window_size-1:], y_hats_rolling_avg, color='grey', alpha=0.6, label=f'{int(window_size/6)}hr Rolling Avg')
    ax[1][1].scatter(
        np.arange(0, len(y_hats), 1),
        y_hats[index_sorted],
        alpha=0.4,
        c="#fc8d59",
        s=1,
    )
    ax[1][1].set_ylabel("life percentage")
    ax[1][1].legend()

    ax[1][1].set_title("validation results")


    # calculate losses
    loss_rmse_val = criterion_rmse(y_hats, y_val)
    loss_mae_val = criterion_mae(y_hats, y_val)
    loss_rmsle_val = criterion_rmsle(y_hats, y_val)
    try:
        r2_val = r2_score(y_val, y_hats)
    except:
        r2_val = 99999

    print_text = f"RMSE = {loss_rmse_val:.3f}\nRMSLE = {loss_rmsle_val:.3f}\nMAE = {loss_mae_val:.3f}\nR2 = {r2_val:.3f}\nR2 {int(window_size/6)}hr avg = {r2_val_avg:.3f}"

    results_dict['loss_mse_val'] = mse_value_val
    results_dict['loss_rmse_val'] = loss_rmse_val.item()
    results_dict['loss_mae_val'] = loss_mae_train.item()
    results_dict['loss_rmsle_val'] = loss_rmsle_val.item()
    results_dict['r2_val'] = r2_val
    results_dict['r2_val_avg'] = r2_val_avg


    # get x and y axis limits
    x_min, x_max = ax[1][1].get_xlim()
    y_min, y_max = ax[1][1].get_ylim()

    ax[1][1].text(
        (x_max - x_min) * 0.95 + x_min,
        y_max - (y_max - y_min) * 0.85,
        print_text,
        fontsize="medium",
        fontweight="semibold",
        verticalalignment="center",
        horizontalalignment="right",
        bbox={"facecolor": "gray", "alpha": 0.2, "pad": 6},
    )

    #### RUL/PERCENT CURVE TRAIN_2
    y_hats = test(net, x_train_2, device, 100)
    index_sorted = np.array(np.argsort(y_train_2, 0).reshape(-1))

    r2_train_2_avg, y_hats_rolling_avg = calc_r2_avg(y_hats, y_train_2, index_sorted, window_size)
    results_dict['r2_train_2_avg'] = r2_train_2_avg

    ax[2][0].plot(np.array(y_train_2)[index_sorted], label="True Life Percentage", alpha=0.5)
    ax[2][0].plot(np.arange(0, len(y_hats), 1)[window_size-1:], y_hats_rolling_avg, color='grey', alpha=0.6, label=f'{int(window_size/6)}hr Rolling Avg')
    ax[2][0].scatter(
        np.arange(0, len(y_hats), 1),
        y_hats[index_sorted],
        label="Predicted Life Percentage",
        alpha=0.4,
        c="#fc8d59",
        s=1,
    )
    ax[2][0].set_ylabel("life percentage")

    ax[2][0].set_title("train_2 results")

    # calculate losses
    loss_rmse_train_2 = criterion_rmse(y_hats, y_train_2)
    loss_mae_train_2 = criterion_mae(y_hats, y_train_2)
    try:
        r2_train_2 = r2_score(y_train_2, y_hats)
    except:
        r2_train_2 = 99999

    print_text = f"RMSE = {loss_rmse_train_2:.3f}\nMAE = {loss_mae_train_2:.3f}\nR2 = {r2_train_2:.3f}\nR2 {int(window_size/6)}hr avg = {r2_train_2_avg:.3f}"

    # get x and y axis limits
    x_min, x_max = ax[2][0].get_xlim()
    y_min, y_max = ax[2][0].get_ylim()

    ax[2][0].text(
        (x_max - x_min) * 0.95 + x_min,
        y_max - (y_max - y_min) * 0.87,
        print_text,
        fontsize="medium",
        fontweight="semibold",
        verticalalignment="center",
        horizontalalignment="right",
        bbox={"facecolor": "gray", "alpha": 0.2, "pad": 6},
    )

    #### RUL/PERCENT CURVE TRAIN_3
    y_hats = test(net, x_train_3, device, 100)
    index_sorted = np.array(np.argsort(y_train_3, 0).reshape(-1))

    r2_train_3_avg, y_hats_rolling_avg = calc_r2_avg(y_hats, y_train_3, index_sorted, window_size)
    results_dict['r2_train_3_avg'] = r2_train_3_avg

    ax[2][1].plot(np.array(y_train_3)[index_sorted], label="True Life Percentage", alpha=0.5)
    ax[2][1].plot(np.arange(0, len(y_hats), 1)[window_size-1:], y_hats_rolling_avg, color='grey', alpha=0.6, label=f'{int(window_size/6)}hr Rolling Avg')
    ax[2][1].scatter(
        np.arange(0, len(y_hats), 1),
        y_hats[index_sorted],
        label="Predicted Life Percentage",
        alpha=0.4,
        c="#fc8d59",
        s=1,
    )
    ax[2][1].set_ylabel("life percentage")

    ax[2][1].set_title("train_3 results")

    # calculate losses
    loss_rmse_train_3 = criterion_rmse(y_hats, y_train_3)
    loss_mae_train_3 = criterion_mae(y_hats, y_train_3)
    try:
        r2_train_3 = r2_score(y_train_3, y_hats)
    except:
        r2_train_3 = 99999

    print_text = f"RMSE = {loss_rmse_train_3:.3f}\nMAE = {loss_mae_train_3:.3f}\nR2 = {r2_train_3:.3f}\nR2 {int(window_size/6)}hr avg = {r2_train_3_avg:.3f}"

    # get x and y axis limits
    x_min, x_max = ax[2][1].get_xlim()
    y_min, y_max = ax[2][1].get_ylim()

    ax[2][1].text(
        (x_max - x_min) * 0.95 + x_min,
        y_max - (y_max - y_min) * 0.87,
        print_text,
        fontsize="medium",
        fontweight="semibold",
        verticalalignment="center",
        horizontalalignment="right",
        bbox={"facecolor": "gray", "alpha": 0.2, "pad": 6},
    )

    plt.ioff()
    if save_pic:
        plt.savefig(save_path / f"{date_time}_{loss_func}_{rnd_seed}.png", format="png", dpi=150)

    if show_pic:
        plt.show()
    else:
        plt.close()

    return (
        epoch_stopped_on,
        results_dict,
    )


def plot_trained_model_results_femto(
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
    save_path,
    loss_func="mse",
    batch_size=99999,
    epochs=99999,
    patience=99999,
    lambda_mod=99999.0,
    learning_rate=99999.0,
    eta=99999.0,
    beta=99999.0,
    n_layers=99999,
    n_units=99999,
    prob_drop=0,
    early_stop_delay=99999,
    save_pic=False,
    show_pic=False,
    rnd_seed=99999,
    data_set='ims',
):
    fig, ax = plt.subplots(4, 3, figsize=(17, 15), constrained_layout=True,)
    net.eval()

    results_dict = {}

    # establish evalutation metrics
    criterion_mae = nn.L1Loss()
    criterion_rmse = RMSELoss()
    criterion_rmsle = RMSLELoss()

    #### SELECTED LOSS FUNCTION
    # in sample loss
    ax[0][0].plot(
        df["epoch"], df["loss"], label="Loss", linewidth=2, alpha=0.4, color="#d73027"
    )

    # out of sample (validation) loss
    ax[0][0].plot(
        df["epoch"], df["val_loss"], label="Val Loss", linewidth=2, color="#d73027"
    )
    try:
        epoch_stopped_on = len(df)-patience
    except:
        epoch_stopped_on = len(df)


    ax[0][0].axvline(
        epoch_stopped_on,
        linestyle="--",
        color="black",
        label="Early Stopping Checkpoint",
        alpha=0.6,
    )

    ax[0][0].legend()
    ax[0][0].set_xlabel("epoch")
    ax[0][0].set_ylabel(f"{loss_func} loss")
    ax[0][0].set_title(f"{loss_func} loss")

    #### MSE LOSS FUNCTION
    # in sample loss
    ax[0][1].plot(
        df["epoch"], df["loss"], label="Loss", linewidth=2, alpha=0.4, color="#d73027"
    )

    # out of sample (validation) loss
    ax[0][1].plot(
        df["epoch"], df["val_loss_mse"], label="Val Loss", linewidth=2, color="#d73027"
    )
    ax[0][1].axvline(
        epoch_stopped_on,
        linestyle="--",
        color="black",
        label="Early Stopping Checkpoint",
        alpha=0.6,
    )

    ax[0][1].set_xlabel("epoch")
    ax[0][1].set_ylabel("mse loss")
    ax[0][1].set_title("mse loss")

    try:
        mse_value_val = df["val_loss_mse"].iloc[epoch_stopped_on - 1]
        mse_value_train = df["loss_mse"].iloc[epoch_stopped_on - 1]
    except:
        mse_value_val = df["val_loss_mse"][0]
        mse_value_train = df["loss_mse"][0]

    # get x and y axis limits
    x_min, x_max = ax[0][1].get_xlim()
    y_min, y_max = ax[0][1].get_ylim()

    print_text = f"MSE = {mse_value_val:.3f}"

    ax[0][1].text(
        (x_max - x_min) * 0.95 + x_min,
        y_max - (y_max - y_min) * 0.1,
        print_text,
        fontsize="medium",
        fontweight="semibold",
        verticalalignment="center",
        horizontalalignment="right",
        bbox={"facecolor": "gray", "alpha": 0.2, "pad": 6},
    )

    #### RUL/PERCENT CURVE TRAIN
    def calc_r2_avg(y_hats, y_val, index_sorted, window_size):
        y_hats_rolling_avg = np.convolve(np.array(y_hats[index_sorted]).reshape(-1), np.ones(window_size), 'valid') / window_size
        try:
            r2_val_avg = r2_score(np.array(y_val)[index_sorted][window_size-1:], y_hats_rolling_avg)
        except:
            r2_val_avg = 99999
        return r2_val_avg, y_hats_rolling_avg


    y_hats = test(net, x_train, device, 100)
    index_sorted = np.array(np.argsort(y_train, 0).reshape(-1))

    ax[0][2].plot(np.array(y_train)[index_sorted], label="True Life Percentage", alpha=0.5)
    ax[0][2].scatter(
        np.arange(0, len(y_hats), 1),
        y_hats[index_sorted],
        label="Predicted Life Percentage",
        alpha=0.4,
        c="#fc8d59",
        s=1,
    )
    ax[0][2].set_ylabel("life percentage")
    ax[0][2].legend(loc='upper left')

    ax[0][2].set_title("train (all) results")


    # calculate losses
    loss_rmse_train = criterion_rmse(y_hats, y_train)
    loss_mae_train = criterion_mae(y_hats, y_train)
    loss_rmsle_train = criterion_rmsle(y_hats, y_train)
    try:
        r2_train = r2_score(y_train, y_hats)
    except:
        r2_train = 99999
    
    print_text = f"RMSE = {loss_rmse_train:.3f}\nRMSLE = {loss_rmsle_train:.3f}\nMAE = {loss_mae_train:.3f}\nR2 = {r2_train:.3f}"

    results_dict['loss_mse_train'] = mse_value_train
    results_dict['loss_rmse_train'] = loss_rmse_train.item()
    results_dict['loss_mae_train'] = loss_mae_train.item()
    results_dict['loss_rmsle_train'] = loss_rmsle_train.item()
    results_dict['r2_train'] = r2_train

    # get x and y axis limits
    x_min, x_max = ax[0][2].get_xlim()
    y_min, y_max = ax[0][2].get_ylim()

    ax[0][2].text(
        (x_max - x_min) * 0.95 + x_min,
        y_max - (y_max - y_min) * 0.87,
        print_text,
        fontsize="medium",
        fontweight="semibold",
        verticalalignment="center",
        horizontalalignment="right",
        bbox={"facecolor": "gray", "alpha": 0.2, "pad": 6},
    )
    ######### BOTTOM TEXT ################
    # bottom text showing model parameters

    bottom_text = f"batch_size = {batch_size},   epochs = {epochs},   patience = {patience} \nlambda_mod = {lambda_mod:.3f},   eta = {eta:.3f},   beta = {beta:.3f}\nn_layers = {n_layers},  n_units = {n_units},   learning_rate = {learning_rate}\nprob_drop = {prob_drop},   early_stop_delay = {early_stop_delay},\nepoch_stopped_on = {epoch_stopped_on} \n "

    ax[3][1].text(
        # x_min + x_max * 0.05,
        # y_min - y_max * 0.15,
        0,
        0.5,
        bottom_text,
        fontsize="large",
        fontweight="normal",
        verticalalignment="top",
        horizontalalignment="left",
    )

    # turnoff axis we won't use
    # https://stackoverflow.com/a/10035974/9214620
    ax[3][1].axis('off')
    ax[3][2].axis('off')

    #### RUL/PERCENT CURVE VAL
    y_hats = test(net, x_val, device, 100)
    index_sorted = np.array(np.argsort(y_val, 0).reshape(-1))

    window_size = 12 # 2 minute rolling avg
    r2_val_avg, y_hats_rolling_avg = calc_r2_avg(y_hats, y_val, index_sorted, window_size)



    ax[1][0].plot(np.array(y_val)[index_sorted], alpha=0.5)
    ax[1][0].plot(np.arange(0, len(y_hats), 1)[window_size-1:], y_hats_rolling_avg, color='grey', alpha=0.6, label=f'{int(window_size/6)}min Rolling Avg')
    ax[1][0].scatter(
        np.arange(0, len(y_hats), 1),
        y_hats[index_sorted],
        alpha=0.4,
        c="#fc8d59",
        s=1,
    )
    ax[1][0].set_ylabel("life percentage")
    ax[1][0].legend(loc='upper left')

    ax[1][0].set_title("validation results")


    # calculate losses
    loss_rmse_val = criterion_rmse(y_hats, y_val)
    loss_mae_val = criterion_mae(y_hats, y_val)
    loss_rmsle_val = criterion_rmsle(y_hats, y_val)
    try:
        r2_val = r2_score(y_val, y_hats)
    except:
        r2_val = 99999

    print_text = f"RMSE = {loss_rmse_val:.3f}\nRMSLE = {loss_rmsle_val:.3f}\nMAE = {loss_mae_val:.3f}\nR2 = {r2_val:.3f}\nR2 {int(window_size/6)}min avg = {r2_val_avg:.3f}"

    results_dict['loss_mse_val'] = mse_value_val
    results_dict['loss_rmse_val'] = loss_rmse_val.item()
    results_dict['loss_mae_val'] = loss_mae_train.item()
    results_dict['loss_rmsle_val'] = loss_rmsle_val.item()
    results_dict['r2_val'] = r2_val
    results_dict['r2_val_avg'] = r2_val_avg


    # get x and y axis limits
    x_min, x_max = ax[1][0].get_xlim()
    y_min, y_max = ax[1][0].get_ylim()

    ax[1][0].text(
        (x_max - x_min) * 0.95 + x_min,
        y_max - (y_max - y_min) * 0.85,
        print_text,
        fontsize="medium",
        fontweight="semibold",
        verticalalignment="center",
        horizontalalignment="right",
        bbox={"facecolor": "gray", "alpha": 0.2, "pad": 6},
    )

    #### RUL/PERCENT CURVE x_train1_1
    y_hats = test(net, x_train1_1, device, 100)
    index_sorted = np.array(np.argsort(y_train1_1, 0).reshape(-1))

    r2_train1_1_avg, y_hats_rolling_avg = calc_r2_avg(y_hats, y_train1_1, index_sorted, window_size)
    results_dict['r2_train1_1_avg'] = r2_train1_1_avg

    ax[1][1].plot(np.array(y_train1_1)[index_sorted], label="True Life Percentage", alpha=0.5)
    ax[1][1].plot(np.arange(0, len(y_hats), 1)[window_size-1:], y_hats_rolling_avg, color='grey', alpha=0.6, label=f'{int(window_size/6)}min Rolling Avg')
    ax[1][1].scatter(
        np.arange(0, len(y_hats), 1),
        y_hats[index_sorted],
        label="Predicted Life Percentage",
        alpha=0.4,
        c="#fc8d59",
        s=1,
    )
    ax[1][1].set_ylabel("life percentage")

    ax[1][1].set_title("bearing1_1 train results")

    # calculate losses
    loss_rmse_train_2 = criterion_rmse(y_hats, y_train1_1)
    loss_mae_train_2 = criterion_mae(y_hats, y_train1_1)
    try:
        r2_train_2 = r2_score(y_train1_1, y_hats)
    except:
        r2_train_2 = 99999

    print_text = f"RMSE = {loss_rmse_train_2:.3f}\nMAE = {loss_mae_train_2:.3f}\nR2 = {r2_train_2:.3f}\nR2 {int(window_size/6)}min avg = {r2_train1_1_avg:.3f}"

    # get x and y axis limits
    x_min, x_max = ax[1][1].get_xlim()
    y_min, y_max = ax[1][1].get_ylim()

    ax[1][1].text(
        (x_max - x_min) * 0.95 + x_min,
        y_max - (y_max - y_min) * 0.87,
        print_text,
        fontsize="medium",
        fontweight="semibold",
        verticalalignment="center",
        horizontalalignment="right",
        bbox={"facecolor": "gray", "alpha": 0.2, "pad": 6},
    )

    #### RUL/PERCENT CURVE x_train2_1
    y_hats = test(net, x_train2_1, device, 100)
    index_sorted = np.array(np.argsort(y_train2_1, 0).reshape(-1))

    r2_train2_1_avg, y_hats_rolling_avg = calc_r2_avg(y_hats, y_train2_1, index_sorted, window_size)
    results_dict['r2_train2_1_avg'] = r2_train2_1_avg

    ax[1][2].plot(np.array(y_train2_1)[index_sorted], label="True Life Percentage", alpha=0.5)
    ax[1][2].plot(np.arange(0, len(y_hats), 1)[window_size-1:], y_hats_rolling_avg, color='grey', alpha=0.6, label=f'{int(window_size/6)}min Rolling Avg')
    ax[1][2].scatter(
        np.arange(0, len(y_hats), 1),
        y_hats[index_sorted],
        label="Predicted Life Percentage",
        alpha=0.4,
        c="#fc8d59",
        s=1,
    )
    ax[1][2].set_ylabel("life percentage")

    ax[1][2].set_title("bearing2_1 train results")

    # calculate losses
    loss_rmse_train_3 = criterion_rmse(y_hats, y_train2_1)
    loss_mae_train_3 = criterion_mae(y_hats, y_train2_1)
    try:
        r2_train_3 = r2_score(y_train2_1, y_hats)
    except:
        r2_train_3 = 99999

    print_text = f"RMSE = {loss_rmse_train_3:.3f}\nMAE = {loss_mae_train_3:.3f}\nR2 = {r2_train_3:.3f}\nR2 {int(window_size/6)}min avg = {r2_train2_1_avg:.3f}"

    # get x and y axis limits
    x_min, x_max = ax[1][2].get_xlim()
    y_min, y_max = ax[1][2].get_ylim()

    ax[1][2].text(
        (x_max - x_min) * 0.95 + x_min,
        y_max - (y_max - y_min) * 0.87,
        print_text,
        fontsize="medium",
        fontweight="semibold",
        verticalalignment="center",
        horizontalalignment="right",
        bbox={"facecolor": "gray", "alpha": 0.2, "pad": 6},
    )

    #### RUL/PERCENT CURVE x_train3_1
    y_hats = test(net, x_train3_1, device, 100)
    index_sorted = np.array(np.argsort(y_train3_1, 0).reshape(-1))

    r2_train3_1_avg, y_hats_rolling_avg = calc_r2_avg(y_hats, y_train3_1, index_sorted, window_size)
    results_dict['r2_train3_1_avg'] = r2_train3_1_avg

    ax[2][0].plot(np.array(y_train3_1)[index_sorted], label="True Life Percentage", alpha=0.5)
    ax[2][0].plot(np.arange(0, len(y_hats), 1)[window_size-1:], y_hats_rolling_avg, color='grey', alpha=0.6, label=f'{int(window_size/6)}min Rolling Avg')
    ax[2][0].scatter(
        np.arange(0, len(y_hats), 1),
        y_hats[index_sorted],
        label="Predicted Life Percentage",
        alpha=0.4,
        c="#fc8d59",
        s=1,
    )
    ax[2][0].set_ylabel("life percentage")

    ax[2][0].set_title("bearing3_1 train results")

    # calculate losses
    loss_rmse_train_3 = criterion_rmse(y_hats, y_train3_1)
    loss_mae_train_3 = criterion_mae(y_hats, y_train3_1)
    try:
        r2_train_3 = r2_score(y_train3_1, y_hats)
    except:
        r2_train_3 = 99999

    print_text = f"RMSE = {loss_rmse_train_3:.3f}\nMAE = {loss_mae_train_3:.3f}\nR2 = {r2_train_3:.3f}\nR2 {int(window_size/6)}min avg = {r2_train3_1_avg:.3f}"

    # get x and y axis limits
    x_min, x_max = ax[2][0].get_xlim()
    y_min, y_max = ax[2][0].get_ylim()

    ax[2][0].text(
        (x_max - x_min) * 0.95 + x_min,
        y_max - (y_max - y_min) * 0.87,
        print_text,
        fontsize="medium",
        fontweight="semibold",
        verticalalignment="center",
        horizontalalignment="right",
        bbox={"facecolor": "gray", "alpha": 0.2, "pad": 6},
    )

    ######## VALIDATION ##########
    #### RUL/PERCENT CURVE x_val1_2
    y_hats = test(net, x_val1_2, device, 100)
    index_sorted = np.array(np.argsort(y_val1_2, 0).reshape(-1))

    r2_val1_2_avg, y_hats_rolling_avg = calc_r2_avg(y_hats, y_val1_2, index_sorted, window_size)
    results_dict['r2_val1_2_avg'] = r2_val1_2_avg

    ax[2][1].plot(np.array(y_val1_2)[index_sorted], label="True Life Percentage", alpha=0.5)
    ax[2][1].plot(np.arange(0, len(y_hats), 1)[window_size-1:], y_hats_rolling_avg, color='grey', alpha=0.6, label=f'{int(window_size/6)}min Rolling Avg')
    ax[2][1].scatter(
        np.arange(0, len(y_hats), 1),
        y_hats[index_sorted],
        label="Predicted Life Percentage",
        alpha=0.4,
        c="#fc8d59",
        s=1,
    )
    ax[2][1].set_ylabel("life percentage")

    ax[2][1].set_title("bearing1_2 val results")

    # calculate losses
    loss_rmse_train_3 = criterion_rmse(y_hats, y_val1_2)
    loss_mae_train_3 = criterion_mae(y_hats, y_val1_2)
    try:
        r2_train_3 = r2_score(y_val1_2, y_hats)
    except:
        r2_train_3 = 99999

    print_text = f"RMSE = {loss_rmse_train_3:.3f}\nMAE = {loss_mae_train_3:.3f}\nR2 = {r2_train_3:.3f}\nR2 {int(window_size/6)}min avg = {r2_val1_2_avg:.3f}"

    # get x and y axis limits
    x_min, x_max = ax[2][1].get_xlim()
    y_min, y_max = ax[2][1].get_ylim()

    ax[2][1].text(
        (x_max - x_min) * 0.95 + x_min,
        y_max - (y_max - y_min) * 0.87,
        print_text,
        fontsize="medium",
        fontweight="semibold",
        verticalalignment="center",
        horizontalalignment="right",
        bbox={"facecolor": "gray", "alpha": 0.2, "pad": 6},
    )

    #### RUL/PERCENT CURVE x_val2_2
    y_hats = test(net, x_val2_2, device, 100)
    index_sorted = np.array(np.argsort(y_val2_2, 0).reshape(-1))

    r2_val2_2_avg, y_hats_rolling_avg = calc_r2_avg(y_hats, y_val2_2, index_sorted, window_size)
    results_dict['r2_val2_2_avg'] = r2_val2_2_avg

    ax[2][2].plot(np.array(y_val2_2)[index_sorted], label="True Life Percentage", alpha=0.5)
    ax[2][2].plot(np.arange(0, len(y_hats), 1)[window_size-1:], y_hats_rolling_avg, color='grey', alpha=0.6, label=f'{int(window_size/6)}min Rolling Avg')
    ax[2][2].scatter(
        np.arange(0, len(y_hats), 1),
        y_hats[index_sorted],
        label="Predicted Life Percentage",
        alpha=0.4,
        c="#fc8d59",
        s=1,
    )
    ax[2][2].set_ylabel("life percentage")

    ax[2][2].set_title("bearing2_2 val results")

    # calculate losses
    loss_rmse_train_3 = criterion_rmse(y_hats, y_val2_2)
    loss_mae_train_3 = criterion_mae(y_hats, y_val2_2)
    try:
        r2_train_3 = r2_score(y_val2_2, y_hats)
    except:
        r2_train_3 = 99999

    print_text = f"RMSE = {loss_rmse_train_3:.3f}\nMAE = {loss_mae_train_3:.3f}\nR2 = {r2_train_3:.3f}\nR2 {int(window_size/6)}min avg = {r2_val2_2_avg:.3f}"

    # get x and y axis limits
    x_min, x_max = ax[2][2].get_xlim()
    y_min, y_max = ax[2][2].get_ylim()

    ax[2][2].text(
        (x_max - x_min) * 0.95 + x_min,
        y_max - (y_max - y_min) * 0.87,
        print_text,
        fontsize="medium",
        fontweight="semibold",
        verticalalignment="center",
        horizontalalignment="right",
        bbox={"facecolor": "gray", "alpha": 0.2, "pad": 6},
    )


    #### RUL/PERCENT CURVE x_val3_2
    y_hats = test(net, x_val3_2, device, 100)
    index_sorted = np.array(np.argsort(y_val3_2, 0).reshape(-1))

    r2_val3_2_avg, y_hats_rolling_avg = calc_r2_avg(y_hats, y_val3_2, index_sorted, window_size)
    results_dict['r2_val3_2_avg'] = r2_val3_2_avg

    ax[3][0].plot(np.array(y_val3_2)[index_sorted], label="True Life Percentage", alpha=0.5)
    ax[3][0].plot(np.arange(0, len(y_hats), 1)[window_size-1:], y_hats_rolling_avg, color='grey', alpha=0.6, label=f'{int(window_size/6)}min Rolling Avg')
    ax[3][0].scatter(
        np.arange(0, len(y_hats), 1),
        y_hats[index_sorted],
        label="Predicted Life Percentage",
        alpha=0.4,
        c="#fc8d59",
        s=1,
    )
    ax[3][0].set_ylabel("life percentage")

    ax[3][0].set_title("bearing3_2 val results")

    # calculate losses
    loss_rmse_train_3 = criterion_rmse(y_hats, y_val3_2)
    loss_mae_train_3 = criterion_mae(y_hats, y_val3_2)
    try:
        r2_train_3 = r2_score(y_val3_2, y_hats)
    except:
        r2_train_3 = 99999

    print_text = f"RMSE = {loss_rmse_train_3:.3f}\nMAE = {loss_mae_train_3:.3f}\nR2 = {r2_train_3:.3f}\nR2 {int(window_size/6)}min avg = {r2_val3_2_avg:.3f}"

    # get x and y axis limits
    x_min, x_max = ax[3][0].get_xlim()
    y_min, y_max = ax[3][0].get_ylim()

    ax[3][0].text(
        (x_max - x_min) * 0.95 + x_min,
        y_max - (y_max - y_min) * 0.87,
        print_text,
        fontsize="medium",
        fontweight="semibold",
        verticalalignment="center",
        horizontalalignment="right",
        bbox={"facecolor": "gray", "alpha": 0.2, "pad": 6},
    )

    plt.ioff()
    if save_pic:
        plt.savefig(save_path / f"{date_time}_{loss_func}_{rnd_seed}.png", format="png", dpi=150, 
        # bbox_inches = "tight"
        )

    if show_pic:
        plt.show()
    else:
        plt.close()

    return (
        epoch_stopped_on,
        results_dict,
    )

