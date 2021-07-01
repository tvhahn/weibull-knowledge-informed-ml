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

    return x, y, xf, yf