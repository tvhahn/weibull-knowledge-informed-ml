import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal, fftpack


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

    plt.show()
