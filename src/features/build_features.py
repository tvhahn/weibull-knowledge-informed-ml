import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal, fftpack
import seaborn as sns


def create_fft(df, y_name='b2_ch3', sample_freq=20480.0, window='hamming', beta=8.0):
    """Create FFT plot from a pandas dataframe of signals

    
    Parameters
    ===========
    df : Pandas dataframe
        Signal that is being normalized

    y_name : str
        Signal name (column name from dataframe) that will be used to generate
        the FFT

    sample_freq : float
        Sampling frequency used to collect the signal

    window : str --> 'hamming' else assumed 'kaiser'
        Chose either the hamming or kaiser windowing function

    beta : float
        Used to determine shape of kaiser function. See scipy documentation
        for additional details. 14 is good to start with.

    Returns
    ===========
    x : ndarray
        Time (likely in seconds). Necessary for plotting time domain signals

    y : ndarray
        Time-domain signal (for example, the acceleration)

    xf : ndarray
        Frequency (likely in Hz). Necessary for plotting the frequency domain

    yf : ndarry
        Amplitude of FFT.

    """

    y = df[y_name].to_numpy(dtype="float64")  # convert to a numpy array
    x = np.arange(0,df.shape[0], dtype='float64') / (sample_freq)

    # parameters for plot
    T = 1.0 / sample_freq  # sample spacing
    N = len(y)  # number of sample points
    
    # do some preprocessing of the current signal
    y_detrend = y - np.mean(y)
    y_detrend = signal.detrend(y_detrend, type="constant")  # detrended signal
    
    if window == 'hamming':
        y_detrend *= np.hamming(N)  # apply a hamming window. Why? https://dsp.stackexchange.com/a/11323
    else:
        y_detrend *= np.kaiser(len(y_detrend), beta)

    # FFT on time domain signal
    yf = fftpack.rfft(y_detrend)
    yf = 2.0 / N * np.abs(yf[: int(N / 2.0)])
    xf = np.linspace(0.0, 1.0 / (2.0 * T), N // 2)/2

    if show_plot:
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
        # axes[0].set_yticklabels([])

        # plot the frequency domain signal
        axes[1].plot(xf, yf, marker="", label="Best model", color=pal[3], linewidth=0.8)
        axes[1].set_title("Frequency Domain", fontdict={"fontweight": "normal"})
        axes[1].set_xlabel("Frequency (Hz)")
        axes[1].set_ylabel("Amplitude")
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        

        # clean up the sub-plots to make everything pretty
        for ax in axes.flatten():
            ax.yaxis.set_tick_params(labelleft=True, which="major")
            ax.grid(False)
            
        # in case you want to save the figure (just uncomment)
        # plt.savefig('time_freq_domains.svg', format='svg', dpi=300,bbox_inches = "tight")
        # plt.savefig('time_freq_domains.png', format='png', dpi=300,bbox_inches = "tight")
        plt.show()
    
    return x, y, xf, yf