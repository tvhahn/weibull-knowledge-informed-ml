import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal, fftpack
from matplotlib import gridspec
import pandas as pd
import logging
from pathlib import Path
import os
import torch
from src.data.data_utils import load_train_test_ims, load_train_test_femto
from src.features.build_features import build_spectrogram_df_ims, create_fft


def create_time_frequency_plot(
    x, y, xf, yf, save_plot=False, save_name="time_freq_domain.svg", dpi=150
):
    """Create a time domain and frequency domain plot.

    Parameters
    ===========
    x : ndarray
        Time (likely in seconds). Necessary for plotting time domain signals

    y : ndarray
        Time-domain signal (for example, the acceleration)

    xf : ndarray
        Frequency (likely in Hz). Necessary for plotting the frequency domain

    yf : ndarry
        Amplitude of FFT.

    save_plot : boolean
        True or False, whether to save the plot to file

    save_name : str
        If saving the plot, what is the name? Can be a string and/or path

    dpi : int
        dpi of saved image, if applicable

    Returns
    ===========
    Saves and/or shows a plot.

    """

    # setup the seaborn plot
    sns.set(font_scale=1.1, style="whitegrid")
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False, sharey=False)
    fig.tight_layout(pad=5.0)

    pal = sns.cubehelix_palette(6, rot=-0.25, light=0.7)  # pick nice color for plot

    # plot time domain signal
    axes[0].plot(x, y, marker="", label="Best model", color=pal[3], linewidth=0.8)
    axes[0].set_title("Time Domain", fontdict={"fontweight": "normal"})
    axes[0].set_xlabel("Time (seconds)")
    axes[0].set_ylabel("Acceleration (g)")

    # plot the frequency domain signal
    axes[1].plot(xf, yf, marker="", label="Best model", color=pal[3], linewidth=0.8)
    axes[1].set_title("Frequency Domain", fontdict={"fontweight": "normal"})
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Amplitude")
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    # clean up the sub-plots to make everything pretty
    for ax in axes.flatten():
        ax.yaxis.set_tick_params(labelleft=True, which="major")
        ax.grid(False)

    if save_plot:
        plt.savefig(save_name, dpi=dpi, bbox_inches="tight")

    else:
        plt.show()


def plot_freq_peaks(
    xf,
    yf,
    max_freq_to_plot=1000,
    peak_height=0.0001,
    peak_distance=100,
    save_plot=False,
    save_name="fft_peaks.png",
    dpi=150,
):
    """Create a frequency domain plot and show peaks with associated
    frequency values.

    Parameters
    ===========

    xf : ndarray
        Frequency (likely in Hz). Necessary for plotting the frequency domain

    yf : ndarry
        Amplitude of FFT.

    max_freq_to_plot : int
        Cuttoff for the

    save_name : str
        If saving the plot, what is the name? Can be a string and/or path

    dpi : int
        dpi of saved image, if applicable

    Returns
    ===========
    Saves and/or shows a plot.

    """

    # select the index number where xf is less than a certain freq
    i = np.where(xf < max_freq_to_plot)[0][-1]
    peak_distance_index = peak_distance * i / max_freq_to_plot

    # setup the seaborn plot
    sns.set(font_scale=1.0, style="whitegrid")
    fig, axes = plt.subplots(1, 1, figsize=(12, 8), sharex=False, sharey=False)
    fig.tight_layout(pad=5.0)

    pal = sns.cubehelix_palette(6, rot=-0.25, light=0.7)  # pick nice color for plot

    # plot the frequency domain signal
    axes.plot(
        xf[:i], yf[:i], marker="", label="Best model", color=pal[3], linewidth=0.8
    )
    axes.set_title("Frequency Domain", fontdict={"fontweight": "normal"})
    axes.set_xlabel("Frequency (Hz)")
    axes.set_ylabel("Amplitude")
    axes.yaxis.set_tick_params(labelleft=True, which="major")
    axes.grid(False)

    peaks, _ = signal.find_peaks(
        yf[:i], height=peak_height, distance=peak_distance_index
    )
    plt.plot(xf[peaks], yf[peaks], "x", color="#d62728", markersize=10)

    for p in peaks:
        axes.text(
            x=xf[p] + max_freq_to_plot / 50.0,
            y=yf[p],
            s=f"{xf[p]:.1f} Hz",
            horizontalalignment="left",
            verticalalignment="center",
            size=12,
            color="#d62728",
            rotation="horizontal",
            weight="normal",
        )

    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    if save_plot:
        plt.savefig(save_name, dpi=dpi, bbox_inches="tight")
    else:
        plt.show()


def plot_spectogram_with_binned(
    df_spec, labels_dict, path_save_name=Path("dummy_folder"), vmax_factor1=0.1, vmax_factor2=0.9, dpi=150, save_plot=True
):
    color_scheme = "inferno"

    days = []
    for i in labels_dict:
        days.append(labels_dict[i][4])

    days = sorted(days)

    fig, ax = plt.subplots(
        1,
        2,
        figsize=(11, 4),
        dpi=dpi,
    )

    vmax_val = np.max(df_spec.to_numpy().flatten())
    ax[0].pcolormesh(
        days,
        df_spec.index,
        df_spec,
        cmap=color_scheme,
        vmax=vmax_val * vmax_factor1,
        shading="auto",
    )

    ax[0].set_yticks([0, 1000, 2000, 3000, 4000, 5000])
    ax[0].set_yticklabels(["", 1000, 2000, 3000, 4000, 5000])
    ax[0].set_ylabel("Frequency (Hz)")
    ax[0].set_xlabel("Runtime (days)")
    ax[0].tick_params(axis="both", which="both", length=0)

    ax[0].text(
        0.01,
        0.99,
        "(a)",
        verticalalignment="top",
        horizontalalignment="left",
        transform=ax[0].transAxes,
        color="white",
        fontsize=12,
    )

    ##### BINED SPECTROGRAM #####
    bucket_size = 500

    samples = df_spec.shape[1]

    df_temp = df_spec.iloc[:10000]
    a = np.array(df_temp)  # make numpy array
    print(np.shape(a))

    # get the y-axis (frequency values)
    y = np.array(df_temp.index)
    y = np.max(y.reshape(-1, bucket_size), axis=1)
    y = list(y.round().astype("int")[::2])
    y.insert(0, 0)
    plt.box(on=None)

    # get the max value for each bucket
    # https://stackoverflow.com/a/15956341/9214620
    max_a = np.max(a.reshape(-1, bucket_size, samples), axis=1)

    ax[1].pcolormesh(
        days,
        np.arange(0, 21),
        max_a,
        cmap=color_scheme,
        vmax=vmax_val * vmax_factor2,
        shading="auto",
    )
    ax[1].set_yticks(np.arange(1.5, 20.5, 2))
    ax[1].set_yticklabels(list(np.arange(2, 21, 2)))
    ax[1].tick_params(axis="both", which="both", length=0)

    ax[1].set_xlabel("Runtime (days)")
    ax[1].set_ylabel("Frequency Bin")

    ax[1].text(
        0.01,
        0.99,
        "(b)",
        verticalalignment="top",
        horizontalalignment="left",
        transform=ax[1].transAxes,
        color="white",
        fontsize=12,
    )

    sns.despine(left=True, bottom=True, right=True)
    if save_plot:
        plt.savefig(path_save_name, dpi=dpi, bbox_inches="tight")
    else:
        plt.show()


def weibull_pdf(t, eta, beta):
    "weibull PDF function"
    return (
        (beta / (eta ** beta))
        * (t ** (beta - 1.0))
        * np.exp(-1.0 * ((t / eta) ** beta))
    )


def weibull_cdf(t, eta, beta):
    "weibull CDF function"
    return 1.0 - np.exp(-1.0 * ((t / eta) ** beta))


def plot_weibull_example(
    beta=2.0, eta=100, path_save_name="weibull_cdf_pdf.svg", dpi=300
):

    pal = sns.cubehelix_palette(6, rot=-0.25, light=0.7)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=False)
    axes[0].title.set_text("Weibull CDF")
    axes[0].set_xlabel("Time (days)", labelpad=10)
    axes[0].set_ylabel("Fraction Failing, F(t)", labelpad=10)
    axes[0].grid(False)

    axes[1].title.set_text("Weibull PDF")
    axes[1].set_xlabel("Time (days)", labelpad=10)
    axes[1].set_ylabel("Probability Density, f(t)", labelpad=10)
    axes[1].grid(False)

    for beta in [2.0]:

        t = np.linspace(0, 300, 1000)
        f = weibull_cdf(t, eta, beta)
        axes[0].plot(t, f, color=pal[5], linewidth=2)
        f = weibull_pdf(t, eta, beta)
        axes[1].plot(t, f, color=pal[5], linewidth=2)
    plt.subplots_adjust(wspace=0.4)
    plt.savefig(path_save_name, dpi=dpi, bbox_inches="tight")


def ims_data_processed_fig(
    folder_data_ims, path_save_name="spectrograms_processed_data_IMS.png", dpi=300, save_plot=True
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

    # y_list
    y_list = [y_train_2, y_train_3, y_val, y_test]

    # x_list
    x_list = [x_train_2, x_train_3, x_val, x_test]

    # y_days_list
    y_days_list = [y_train_days_2, y_train_days_3, y_val_days, y_test_days]

    val_max_list = [0.3, 0.3, 0.3, 0.3]

    color_scheme = "inferno"

    fig = plt.figure(
        figsize=(11, 8), dpi=dpi
    )
    gs = gridspec.GridSpec(2, 2)

    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[0, 1])
    ax3 = plt.subplot(gs[1, 0])
    ax4 = plt.subplot(gs[1, 1])
    gs.update(wspace=0.2, hspace=0.3)

    ## General Formatting ##
    # create list of axis elements
    axes_list = [ax1, ax2, ax3, ax4]

    for ax in axes_list:
        ax.grid(b=None)

    ###### TEST DATA #####
    plt.rcParams["axes.titlepad"] = 7

    # secondary axis title list
    ax_title_list = [
        "(a)" + " Train Data (run 2, bearing 1)",
        "(b)" + " Train Data (run 3, bearing 3)",
        "(c)" + " Val Data (run 1, bearing 3)",
        "(d)" + " Test Data (run 1, bearing 4)",
    ]

    counter = 0
    for ax, ax_title, y_temp, x_temp, y_days, val_max in zip(
        axes_list, ax_title_list, y_list, x_list, y_days_list, val_max_list
    ):

        index_sorted = np.array(np.argsort(y_temp, 0).reshape(-1))

        time_array = np.sort(y_days[:, -1])

        index_new = np.arange(0, len(time_array), int(len(time_array) / 3) - 1)

        labels_new = [f"{i:.1f}" for i in time_array[index_new]]
        labels_new[0] = "0"

        ax.pcolormesh(
            x_temp[index_sorted].T,
            cmap=color_scheme,
            vmax=val_max,
        )

        ax.set_xticks(index_new)
        ax.set_xticklabels(
            labels_new,
        )

        ax.text(
            0.02,
            0.97,
            ax_title,
            verticalalignment="top",
            horizontalalignment="left",
            transform=ax.transAxes,
            color="white",
            fontsize=12,
        )

        if counter == 0:
            ax.set_xticks(index_new)
            ax.set_xticklabels(
                labels_new,
            )
            ax.set_yticks(np.arange(3.5, 20.5, 4))
            ax.set_yticklabels(list(np.arange(4, 21, 4)))
            ax.set_ylabel("Frequency Bin")
            ax.set_xlabel("Runtime (days)")
        else:
            ax.set_yticklabels([])

        if counter != 0:
            ax.set_yticklabels([])

        counter += 1

    sns.despine(left=True, bottom=True, right=True)
    if save_plot:
        plt.savefig(path_save_name, bbox_inches="tight")
    else:
        plt.show()


def femto_data_processed_fig(
    folder_data_femto,
    path_save_name="spectrograms_processed_data_FEMTO.png",
    dpi=300,
    vmax_val=0.15,
    save_plot=True,
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
        "(a)" + " Train Data (Bearing1_1)",
        "(b)" + " Train Data (Bearing2_1)",
        "(c)" + " Train Data (Bearing3_1)",
        "(d)" + " Val Data (Bearing1_2)",
        "(e)" + " Val Data (Bearing2_2)",
        "(f)" + " Val Data (Bearing3_2)",
        "(g)" + " Test Data (Bearing1_3)",
        "(h)" + " Test Data (Bearing2_3)",
        "(i)" + " Test Data (Bearing3_3)",
    ]

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
    gs.update(wspace=0.15, hspace=0.3)

    # create list of axis elements
    axes_list = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]

    color_scheme = "inferno"
    counter = 0
    for ax, ax_title, y_temp, x_temp, y_days in zip(
        axes_list, ax_title_list, y_list, x_list, y_days_list
    ):

        index_sorted = np.array(np.argsort(y_temp, 0).reshape(-1))

        time_array = np.sort(y_days[:, -1])

        index_new = np.arange(0, len(time_array), int(len(time_array) / 3) - 1)

        labels_new = [f"{i*24:.1f}" for i in time_array[index_new]]
        # change first value to '0'
        labels_new[0] = "0"

        ax.pcolormesh(
            x_temp[index_sorted].T,
            cmap=color_scheme,
            vmax=vmax_val,
        )

        ax.set_xticks(index_new)
        ax.set_xticklabels(
            labels_new,
        )

        ax.text(
            0.02,
            0.97,
            ax_title,
            verticalalignment="top",
            horizontalalignment="left",
            transform=ax.transAxes,
            color="white",
            fontsize=12,
        )

        if counter == 0:
            ax.set_xticks(index_new)
            ax.set_xticklabels(
                labels_new,
            )
            ax.set_yticks(np.arange(3.5, 20.5, 4))
            ax.set_yticklabels(list(np.arange(4, 21, 4)))
            ax.set_ylabel("Frequency Bin")
            ax.set_xlabel("Runtime (hours)")
        else:
            ax.set_yticklabels([])

        if counter != 0:
            ax.set_yticklabels([])

        counter += 1

    sns.despine(left=True, bottom=True, right=True)
    if save_plot:
        plt.savefig(path_save_name, dpi=dpi, bbox_inches="tight")
    else:
        plt.show()


def main():
    logger = logging.getLogger(__name__)
    logger.info("making figures from results")

    path_raw_data = root_dir / "data/raw/IMS/"
    path_save_loc = root_dir / "reports/figures/"
    folder_data_ims = root_dir / "data/processed/IMS/"
    folder_data_femto = root_dir / "data/processed/FEMTO/"

    folder_2nd = path_raw_data / "2nd_test"
    date_list2 = sorted(os.listdir(folder_2nd))
    col_names = ["b1_ch1", "b2_ch2", "b3_ch3", "b4_ch4"]
    df_spec, labels_dict = build_spectrogram_df_ims(
        folder_2nd,
        date_list2,
        channel_name="b1_ch1",
        start_time=date_list2[0],
        col_names=col_names,
    )

    ######################
    # EXAMPLE SPECTROGRAM AND FEATURE PLOT
    plot_spectogram_with_binned(
        df_spec,
        labels_dict,
        path_save_loc / "spectrogram_with_binned_example.png",
        vmax_factor1=0.08,
        vmax_factor2=0.5,
        dpi=150,
    )

    sns.set(font_scale=0.8, style="whitegrid", font="DejaVu Sans")

    ######################
    # WEIBULL CDF/PDF PLOT
    plot_weibull_example(
        beta=2.0,
        eta=100,
        path_save_name=path_save_loc / "weibull_cdf_pdf_example.pdf",
        dpi=300,
    )

    ######################
    # TIME-DOMAIN, FREQ DOMAIN PLOT
    folder_1st = path_raw_data / "1st_test"
    col_names = [
        "b1_ch1",
        "b1_ch2",
        "b2_ch3",
        "b2_ch4",
        "b3_ch5",
        "b3_ch6",
        "b4_ch7",
        "b4_ch8",
    ]

    df = pd.read_csv(folder_1st / "2003.10.22.12.06.24", sep="\t", names=col_names)
    x, y, xf, yf = create_fft(
        df,
        y_name="b1_ch2",
        sample_freq=20480.0,
        window="kaiser",
        beta=3,
    )
    create_time_frequency_plot(
        x,
        y,
        xf,
        yf,
        save_plot=True,
        save_name=path_save_loc / "time_freq_domain_example.pdf",
    )

    ######################
    # IMS PROCESSED DATA FIGURE
    ims_data_processed_fig(
        folder_data_ims,
        path_save_name=path_save_loc / "ims_spectrograms_processed_data.png",
        dpi=300,
    )

    ######################
    # PRONOSTIA (FEMTO) PROCESSED DATA FIGURE
    femto_data_processed_fig(
        folder_data_femto,
        path_save_name=path_save_loc / "femto_spectrograms_processed_data.png",
        dpi=300,
        vmax_val=0.15,
    )


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    root_dir = Path(__file__).resolve().parents[2]

    main()
