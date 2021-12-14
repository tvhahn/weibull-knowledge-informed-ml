from pathlib import Path
import seaborn as sns

# check if "scratch" path exists in the home directory
# if it does, assume we are on HPC
scratch_path = Path.home() / "scratch"
if scratch_path.exists():
    print("Assume on HPC. Using Agg backend for mpl.")
    import matplotlib
    # run matplotlib without display
    # https://stackoverflow.com/a/4706614/9214620
    matplotlib.use("Agg")
else:
    print("Assume on local compute")
    import matplotlib

import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import pandas as pd
import click
import logging
from src.data.data_utils import load_train_test_ims, load_train_test_femto
from src.models.loss import (
    RMSELoss,
    RMSLELoss,
    WeibullLossRMSE,
    WeibullLossRMSLE,
    WeibullLossMSE,
)
from src.models.utils import test
import torch
import torch.nn as nn
from sklearn.metrics import r2_score
from shutil import copyfile
import os


"""
Functions and scripts to visualize the results.
"""


def loss_function_percentage_fig(
    path_ims_count_csv, path_femto_count_csv, path_save_name="loss_function_percentages.pdf", dpi=300, figsize=(14, 7), save_plot=True
):
    """Visualize the percentage of each loss function"""
    dfi = pd.read_csv(path_ims_count_csv)
    dfp = pd.read_csv(path_femto_count_csv)

    sns.set(font_scale=1.0, style="whitegrid", font="DejaVu Sans") # proper formatting

    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=figsize,
        dpi=dpi,
    )

    title_list = [
        r"$\bf{(a)}$" + " IMS Most Successful Loss Function, by Percentage",
        r"$\bf{(b)}$" + " PRONOSTIA Most Successful Loss Function, by Percentage",
    ]
    df_list = [dfi, dfp]

    for ax, df, title in zip(axes.flat, df_list, title_list):

        ax = sns.barplot(x="percent", y="loss_func", data=df, palette="Blues_d", ax=ax)

        for p in ax.patches:
            # help from https://stackoverflow.com/a/56780852/9214620
            space = 0.5
            _x = p.get_x() + p.get_width() + float(space)
            _y = p.get_y() + p.get_height() / 2
            value = p.get_width()
            ax.text(
                _x,
                _y,
                f"{value:.1f} %",
                ha="left",
                va="center",
                weight="semibold",
                size=12,
            )

        ax.spines["bottom"].set_visible(True)
        ax.set_ylabel("")
        ax.set_xlabel("")
        ax.grid(alpha=0.7, linewidth=1, axis="x")
        ax.set_xticks([0])
        ax.set_xticklabels([], alpha=0)
        ax.set_title(title, fontsize=12, loc="right")

    plt.subplots_adjust(wspace=0.8)
    sns.despine(left=True, bottom=True)
    if save_plot:
        plt.savefig(path_save_name, dpi=dpi, bbox_inches="tight")
        plt.cla()
        plt.close()
    else:
        plt.show()


def early_stop_distribution_fig(
    path_ims_results, path_femto_results, path_save_name="epoch_stop_dist.pdf", dpi=300, figsize=(12, 5), save_plot=True
):
    """Visualize the distribution of when early stopping occured, by loss function"""
    dfi = pd.read_csv(path_ims_results)
    dfp = pd.read_csv(path_femto_results)

    sns.set(font_scale=1.0, style="whitegrid", font="DejaVu Sans") # proper formatting

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize)

    title_list = [
        r"$\bf{(a)}$" + " IMS Distribution of Early Stopping Times",
        r"$\bf{(b)}$" + " PRONOSTIA Distribution of Early Stopping Times",
    ]
    df_list = [dfi, dfp]

    for i, (ax, df, title) in enumerate(zip(axes.flat, df_list, title_list)):

        if i == 0:
            df = df[df["epoch_stopped_on"] < 250]
        else:
            df = df[df["epoch_stopped_on"] < 1250]

        ax = sns.violinplot(
            x="epoch_stopped_on",
            y="weibull_loss",
            data=df,
            scale="width",
            inner=None,
            linewidth=2,
            color="white",
            saturation=1,
            cut=0,
            orient="h",
            zorder=0,
            width=0.8,
            ax=ax,
        )

        # strip plot
        ax = sns.stripplot(
            x="epoch_stopped_on",
            y="weibull_loss",
            data=df,
            size=6,
            jitter=0.2,
            color="black",
            linewidth=0.5,
            marker="o",
            edgecolor=None,
            alpha=0.1,
            zorder=4,
            orient="h",
            ax=ax,
        )
        if i == 0:
            ax.set_yticklabels(
                ("Traditional Loss\nFunctions", "Weibull Loss\nFunctions"), fontsize=12
            )
        else:
            ax.set_yticklabels([])

        ax.grid(axis="x", alpha=0.5)

        ax.yaxis.label.set_visible(False)
        ax.tick_params(axis="x", labelsize=12)
        ax.set_xlabel("Early Stopping Epoch", labelpad=10, fontsize=12)
        ax.set_title(title, fontsize=12, loc="left")
    sns.despine(bottom=True, left=True)
    if save_plot:
        plt.savefig(path_save_name, dpi=dpi, bbox_inches="tight")
        plt.cla()
        plt.close()
    else:
        plt.show()


def loss_function_correlation_fig(
    path_ims_corr_csv, path_femto_corr_csv, path_save_name="correlations.pdf", figsize=(10, 12), dpi=300, save_plot=True
):
    dfi = pd.read_csv(path_ims_corr_csv)
    dfp = pd.read_csv(path_femto_corr_csv)

    sns.set(font_scale=1.0, style="whitegrid", font="DejaVu Sans") # proper formatting

    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=figsize,
    )

    title_list = [
        r"$\bf{(a)}$" + " IMS Correlation of Loss Functions with Test $R^2$",
        r"$\bf{(b)}$" + " PRONOSTIA Correlation of Loss Functions with Test $R^2$",
    ]

    df_list = [dfi, dfp]

    for top_bot_i, (ax, df, title) in enumerate(zip(axes.flat, df_list, title_list)):

        df = df.dropna(axis=0)

        ax = sns.barplot(x="loss_func", y="corr", ax=ax, palette="rocket", data=df)
        ax.set_xticklabels(
            ax.get_xticklabels(),
            rotation=90,
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        # ax.axes.get_yaxis().set_visible(False) # hide y-axis
        ax.grid(alpha=0.7, linewidth=1, axis="y")
        ax.xaxis.set_label_text("", size="large", weight="semibold")
        ax.yaxis.set_label_text("", size="large", weight="semibold")
        ax.set_yticks([0])
        ax.set_yticklabels([], alpha=0)
        ax.set_xticklabels([], alpha=0)

        for i, p in enumerate(ax.patches):
            # help from https://stackoverflow.com/a/56780852/9214620
            space = np.absolute(df["corr"].max() * 0.1)

            value = p.get_height()
            if value >= 0:
                _y = p.get_y() + p.get_height() + float(space)
                _x = p.get_x() + p.get_width() / 2
                ax.text(
                    _x,
                    _y,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    weight="semibold",
                    size=15,
                )
                ax.text(
                    _x + 0.2,
                    -0.01,
                    df["loss_func"][i],
                    ha="right",
                    va="top",
                    weight="normal",
                    multialignment="right",
                    size=15,
                    rotation=65,
                )

            else:
                _y = p.get_y() + p.get_height() - float(space)
                _x = p.get_x() + p.get_width() / 2
                ax.text(
                    _x,
                    _y,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    weight="semibold",
                    size=15,
                )
                ax.text(
                    _x - 0.2,
                    0.01,
                    df["loss_func"][i],
                    ha="left",
                    va="bottom",
                    weight="normal",
                    multialignment="left",
                    size=15,
                    rotation=65,
                )
                
        if top_bot_i == 1:
            text_bracket = "Not statistically significant"
            ax.annotate(text_bracket, xy=(3.5, -0.13),  xycoords='data',
                xytext=(3.5, -0.18), textcoords='data', font="DejaVu Sans",
                arrowprops=dict(color='.15', arrowstyle='-[, widthB=4.8, lengthB=1.0', lw=2.0),
                horizontalalignment='center', verticalalignment='bottom',)

        plt.rcParams["axes.titlepad"] = 20
        ax.set_title(title, loc="left", size=15)

    plt.subplots_adjust(hspace=0.3)
    if save_plot:
        plt.savefig(path_save_name, dpi=dpi, bbox_inches="tight")
        plt.cla()
        plt.close()
    else:
        plt.show()


def calc_r2_avg(y_hats, y_val, index_sorted, window_size):
    """Calculate rolling average for RUL"""
    y_hats_rolling_avg = (
        np.convolve(
            np.array(y_hats[index_sorted]).reshape(-1),
            np.ones(window_size),
            "valid",
        )
        / window_size
    )
    r2_val_avg = r2_score(
        np.array(y_val)[index_sorted][window_size - 1 :], y_hats_rolling_avg
    )
    return r2_val_avg, y_hats_rolling_avg


def femto_results_rul_fig(
    path_top_model_folder, top_model_name, folder_data_femto, path_save_name, dpi=300, save_plot=True
):
    # load data
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
    ) = load_train_test_femto(folder_data_femto)

    y_train1_1_days = torch.reshape(y_train1_1[:, 0], (-1, 1))
    y_train2_1_days = torch.reshape(y_train2_1[:, 0], (-1, 1))
    y_train3_1_days = torch.reshape(y_train3_1[:, 0], (-1, 1))
    y_val1_2_days = torch.reshape(y_val1_2[:, 0], (-1, 1))
    y_val2_2_days = torch.reshape(y_val2_2[:, 0], (-1, 1))
    y_val3_2_days = torch.reshape(y_val3_2[:, 0], (-1, 1))
    y_test1_3_days = torch.reshape(y_test1_3[:, 0], (-1, 1))
    y_test2_3_days = torch.reshape(y_test2_3[:, 0], (-1, 1))
    y_test3_3_days = torch.reshape(y_test3_3[:, 0], (-1, 1))

    y_train = torch.reshape(y_train[:, 1], (-1, 1))
    y_val = torch.reshape(y_val[:, 1], (-1, 1))
    y_test = torch.reshape(y_test[:, 1], (-1, 1))

    y_train1_1 = torch.reshape(y_train1_1[:, 1], (-1, 1))
    y_train2_1 = torch.reshape(y_train2_1[:, 1], (-1, 1))
    y_train3_1 = torch.reshape(y_train3_1[:, 1], (-1, 1))
    y_val1_2 = torch.reshape(y_val1_2[:, 1], (-1, 1))
    y_val2_2 = torch.reshape(y_val2_2[:, 1], (-1, 1))
    y_val3_2 = torch.reshape(y_val3_2[:, 1], (-1, 1))
    y_test1_3 = torch.reshape(y_test1_3[:, 1], (-1, 1))
    y_test2_3 = torch.reshape(y_test2_3[:, 1], (-1, 1))
    y_test3_3 = torch.reshape(y_test3_3[:, 1], (-1, 1))

    # select device to run neural net on
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Running on GPU")
    else:
        device = torch.device("cpu")
        print("Running on CPU")

    criterion_rmse = RMSELoss()

    net = torch.load(path_top_model_folder / top_model_name, map_location=device)

    # y_list
    y_list = [
        y_train1_1,
        y_train2_1,
        y_train3_1,
        y_val1_2,
        y_val2_2,
        y_val3_2,
        y_test1_3,
        y_test2_3,
        y_test3_3,
    ]

    # x_list
    x_list = [
        x_train1_1,
        x_train2_1,
        x_train3_1,
        x_val1_2,
        x_val2_2,
        x_val3_2,
        x_test1_3,
        x_test2_3,
        x_test3_3,
    ]

    # y_days_list
    y_days_list = [
        y_train1_1_days,
        y_train2_1_days,
        y_train3_1_days,
        y_val1_2_days,
        y_val2_2_days,
        y_val3_2_days,
        y_test1_3_days,
        y_test2_3_days,
        y_test3_3_days,
    ]

    ax_title_list = [
        r"$\bf{(a)}$" + " Train Results (Bearing1_1)",
        r"$\bf{(b)}$" + " Train Results (Bearing2_1)",
        r"$\bf{(c)}$" + " Train Results (Bearing3_1)",
        r"$\bf{(d)}$" + " Val Results (Bearing1_2)",
        r"$\bf{(e)}$" + " Val Results (Bearing2_2)",
        r"$\bf{(f)}$" + " Val Results (Bearing3_2)",
        r"$\bf{(g)}$" + " Test Results (Bearing1_3)",
        r"$\bf{(h)}$" + " Test Results (Bearing2_3)",
        r"$\bf{(i)}$" + " Test Results (Bearing3_3)",
    ]

    ###### CREATE FIGURE #####
    # color blind colors, from https://bit.ly/3qJ6LYL
    # [#d73027, #fc8d59, #fee090, #4575b4]
    # [redish, orangeish, yellowish, blueish]

    sns.set(
        font_scale=1.0,
        style="whitegrid",
    )

    # establish subplot axes
    # helpful matplotlib guide:
    # https://matplotlib.org/2.0.2/users/gridspec.html
    fig = plt.figure(figsize=(14, 12), dpi=dpi)
    gs = gridspec.GridSpec(3, 3)
    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[0, 1])
    ax3 = plt.subplot(gs[0, 2])
    ax4 = plt.subplot(gs[1, 0])
    ax5 = plt.subplot(gs[1, 1])
    ax6 = plt.subplot(gs[1, 2])
    ax7 = plt.subplot(gs[2, 0])
    ax8 = plt.subplot(gs[2, 1])
    ax9 = plt.subplot(gs[2, 2])
    gs.update(wspace=0.15, hspace=0.4)

    ## General Formatting ##
    # create list of axis elements
    axes_list = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]

    for ax in axes_list:
        ax.grid(b=None)

    plt.rcParams["axes.titlepad"] = 7

    ###### TEST DATA #####
    net.eval()

    counter = 0
    for ax, y_temp, x_temp, y_days, ax_title in zip(
        axes_list, y_list, x_list, y_days_list, ax_title_list
    ):

        y_hats = test(net, x_temp, device, 100)
        index_sorted = np.array(np.argsort(y_temp, 0).reshape(-1))

        # build rolling average
        window_size = 12  # 2 minute rolling avg
        r2_test_avg, y_hats_rolling_avg = calc_r2_avg(
            y_hats, y_temp, index_sorted, window_size
        )

        loss_rmse_test = criterion_rmse(y_hats, y_temp)
        r2_test = r2_score(y_temp, y_hats)

        ax.plot(
            np.array(y_temp)[index_sorted] * 100,
            label="True Life Percentage",
            alpha=1,
            color="#4575b4",
            linewidth=1,
            zorder=0,
        )
        ax.scatter(
            np.arange(0, len(y_hats), 1),
            y_hats[index_sorted] * 100,
            label="Predicted Life Percentage",
            alpha=0.4,
            c="grey",
            edgecolors="none",
            s=2,
        )

        ax.plot(
            np.arange(0, len(y_hats), 1)[window_size - 1 :],
            y_hats_rolling_avg * 100,
            color="#d73027",
            alpha=1,
            label=f"{int(window_size/6)}min Rolling Avg",
            linewidth=0.5,
        )

        print_text = f"RMSE = {loss_rmse_test:.3f}\n$R^2$ = {r2_test:.3f}"

        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()

        ax.text(
            (x_max - x_min) * 0.03 + x_min,
            y_max - (y_max - y_min) * 0.05,
            print_text,
            fontsize=9,
            fontweight="normal",
            verticalalignment="top",
            horizontalalignment="left",
            bbox={"facecolor": "gray", "alpha": 0.0, "pad": 6},
        )

        index_new = np.arange(0, len(y_hats), int(len(y_hats) / 3) - 1)

        y_days_temp = np.array(y_days)
        y_days_temp = np.reshape(y_days_temp, np.shape(y_days_temp)[0])[index_sorted]

        labels_new = [f"{i*24:.1f}" for i in y_days_temp[index_new]]
        # change first value to '0'
        labels_new[0] = "0"

        ax.set_xticks(index_new)
        ax.set_xticklabels(
            labels_new,
        )
        ax.set_title(ax_title, loc="left")

        if counter == 0:
            ax.set_xlabel("Runtime (hours)")
            ax.set_ylabel("Life Percentage")
            ax.legend(loc="lower right", fontsize=10)

        if counter != 0:
            ax.set_yticklabels([])

        counter += 1
    if save_plot:
        plt.savefig(path_save_name, dpi=dpi, bbox_inches="tight")
        plt.cla()
        plt.close()
    else:
        plt.show()


def ims_results_rul_fig(
    path_top_model_folder, top_model_name, folder_data_ims, path_save_name, dpi=300, save_plot=True
):
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
    ) = load_train_test_ims(folder_data_ims)

    y_train_days = torch.reshape(y_train[:, 0], (-1, 1))
    y_val_days = torch.reshape(y_val[:, 0], (-1, 1))
    y_test_days = torch.reshape(y_test[:, 0], (-1, 1))

    y_train_days_2 = torch.reshape(y_train_2[:, 0], (-1, 1))
    y_train_days_3 = torch.reshape(y_train_3[:, 0], (-1, 1))

    y_train = torch.reshape(y_train[:, 1], (-1, 1))
    y_val = torch.reshape(y_val[:, 1], (-1, 1))
    y_test = torch.reshape(y_test[:, 1], (-1, 1))

    y_train_2 = torch.reshape(y_train_2[:, 1], (-1, 1))
    y_train_3 = torch.reshape(y_train_3[:, 1], (-1, 1))

    # select device to run neural net on
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Running on GPU")
    else:
        device = torch.device("cpu")
        print("Running on CPU")

    criterion_rmse = RMSELoss()

    net = torch.load(path_top_model_folder / top_model_name, map_location=device)

    ###### CREATE FIGURE #####
    # color blind colors, from https://bit.ly/3qJ6LYL
    # [#d73027, #fc8d59, #fee090, #4575b4]
    # [redish, orangeish, yellowish, blueish]

    sns.set(
        font_scale=1.0,
        style="whitegrid",
    )

    # establish subplot axes
    # helpful matplotlib guide:
    # https://matplotlib.org/2.0.2/users/gridspec.html
    fig = plt.figure(figsize=(11, 8), dpi=dpi)
    gs = gridspec.GridSpec(2, 2)

    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[0, 1])
    ax3 = plt.subplot(gs[1, 0])
    ax4 = plt.subplot(gs[1, 1])
    gs.update(wspace=0.2, hspace=0.4)

    ## General Formatting ##
    # create list of axis elements
    axes_list = [ax1, ax2, ax3, ax4]

    for ax in axes_list:
        ax.grid(b=None)

    ###### TEST DATA #####
    net.eval()

    plt.rcParams["axes.titlepad"] = 7

    # secondary axis title list
    ax_title_list = [
        r"$\bf{(a)}$" + " Train Results (run 2, bearing 1)",
        r"$\bf{(b)}$" + " Train Results (run 3, bearing 3)",
        r"$\bf{(c)}$" + " Val Results (run 1, bearing 3)",
        r"$\bf{(d)}$" + " Test Results (run 1, bearing 4)",
    ]

    # secondary axis counter
    counter = 0
    for ax, y_temp, x_temp, y_days, ax_title in zip(
        [ax1, ax2, ax3, ax4],
        [y_train_2, y_train_3, y_val, y_test],
        [x_train_2, x_train_3, x_val, x_test],
        [y_train_days_2, y_train_days_3, y_val_days, y_test_days],
        ax_title_list,
    ):

        y_hats = test(net, x_temp, device, 100)
        index_sorted = np.array(np.argsort(y_temp, 0).reshape(-1))

        # build rolling average
        window_size = 12  # 2 hour rolling avg
        r2_test_avg, y_hats_rolling_avg = calc_r2_avg(
            y_hats, y_temp, index_sorted, window_size
        )

        loss_rmse_test = criterion_rmse(y_hats, y_temp)
        r2_test = r2_score(y_temp, y_hats)

        ax.plot(
            np.array(y_temp)[index_sorted] * 100,
            label="True Life Percentage",
            alpha=1,
            color="#4575b4",
            linewidth=1,
            zorder=0,
        )
        ax.scatter(
            np.arange(0, len(y_hats), 1),
            y_hats[index_sorted] * 100,
            label="Predicted Life Percentage",
            alpha=0.4,
            c="grey",
            edgecolors="none",
            s=2,
        )

        ax.plot(
            np.arange(0, len(y_hats), 1)[window_size - 1 :],
            y_hats_rolling_avg * 100,
            color="#d73027",
            alpha=1,
            label=f"{int(window_size/6)}hr Rolling Avg",
            linewidth=0.5,
        )

        print_text = f"RMSE = {loss_rmse_test:.3f}\n$R^2$ = {r2_test:.3f}"

        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()

        ax.text(
            (x_max - x_min) * 0.03 + x_min,
            y_max - (y_max - y_min) * 0.05,
            print_text,
            fontsize=9,
            fontweight="normal",
            verticalalignment="top",
            horizontalalignment="left",
            bbox={"facecolor": "gray", "alpha": 0.0, "pad": 6},
        )

        index_new = np.arange(0, len(y_hats), int(len(y_hats) / 3) - 1)

        y_days_temp = np.array(y_days)
        y_days_temp = np.reshape(y_days_temp, np.shape(y_days_temp)[0])[index_sorted]

        labels_new = [f"{i:.1f}" for i in y_days_temp[index_new]]
        # change first value to '0'
        labels_new[0] = "0"

        ax.set_xticks(index_new)
        ax.set_xticklabels(
            labels_new,
        )
        ax.set_title(ax_title, loc="left")

        if counter == 0:
            ax.set_xlabel("Runtime (days)")
            ax.set_ylabel("Life Percentage")
            ax.legend(loc="lower right", fontsize=10)

        if counter != 0:
            ax.set_yticklabels([])

        counter += 1

    if save_plot:
        plt.savefig(path_save_name, dpi=dpi, bbox_inches="tight")
        plt.cla()
        plt.close()
    else:
        plt.show()


def main():
    logger = logging.getLogger(__name__)
    logger.info("making figures from results")

    sns.set(font_scale=1.0, style="whitegrid", font="DejaVu Sans")

    # csv result locations
    path_results = root_dir / "models/final/"
    path_save_loc = root_dir / "reports/figures/"

    loss_function_percentage_fig(
        path_results / "ims_count_results.csv",
        path_results / "femto_count_results.csv",
        path_save_loc / "loss_function_percentages.pdf",
    )

    loss_function_correlation_fig(
        path_results / "ims_correlation_results.csv",
        path_results / "femto_correlation_results.csv",
        path_save_loc / "correlations.pdf",
    )

    early_stop_distribution_fig(
        path_results / "ims_results_filtered.csv",
        path_results / "femto_results_filtered.csv",
        path_save_loc / "epoch_stop_dist.pdf",
        dpi=150,
    )

    # create the plots of the top models with predictions on them
    folder_data_ims = root_dir / "data/processed/IMS/"
    folder_data_femto = root_dir / "data/processed/FEMTO/"

    # need model for loading checkpoint
    copyfile(path_results / "top_models_femto/model.py", "model.py")

    # make PRONOSTIA (FEMTO) rul plots
    df_top = pd.read_csv(root_dir / "models/final/femto_results_filtered.csv")
    model_name = df_top["model_checkpoint_name"][0]  # select top model

    femto_results_rul_fig(
        root_dir / "models/final/top_models_femto",
        model_name,
        folder_data_femto,
        path_save_loc / "femto_rul_results.pdf",
        dpi=150,
    )

    os.remove("model.py")  # delete model.py that was copied to root dir

    # need model for loading checkpoint
    copyfile(path_results / "top_models_ims/model.py", "model.py")

    # make IMS rul plots
    df_top = pd.read_csv(root_dir / "models/final/ims_results_filtered.csv")
    model_name = df_top["model_checkpoint_name"][0]  # select top model
    ims_results_rul_fig(
        root_dir / "models/final/top_models_ims",
        model_name,
        folder_data_ims,
        path_save_loc / "ims_rul_results.pdf",
        dpi=150,
    )

    os.remove("model.py")  # delete model.py that was copied to root dir


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    root_dir = Path(__file__).resolve().parents[2]

    main()
