import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal, fftpack
import seaborn as sns
import time
import datetime


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


def build_spectrogram_df(folder, date_list, channel_name='b3_ch5', start_time='2003.10.22.12.06.24', col_day_increment=False,
                         col_names = ['b1_ch1', 'b1_ch2', 'b2_ch3', 'b2_ch4', 'b3_ch5', 'b3_ch6', 'b4_ch7', 'b4_ch8']):
    '''function that builds the spectrogram data'''
    
    # convert start_time to unix timestamp
    start_time = time.mktime(datetime.datetime.strptime(start_time, "%Y.%m.%d.%H.%M.%S").timetuple())

    # instantiate dataframe for the spectrogram
    dft = pd.DataFrame()
       
    # dictionary to store any labels
    labels_dict = {}

    # iterate through each date that samples were taken
    # date_list should be sorted from earliest to latest
    for i, sample_name in enumerate(date_list):
        # convert sample_name to unix timestamp
        unix_timestamp = time.mktime(datetime.datetime.strptime(sample_name, "%Y.%m.%d.%H.%M.%S").timetuple())
        date_nice_format = datetime.datetime.fromtimestamp(unix_timestamp).strftime('%Y-%m-%d %H:%M:%S') # reformat date

        # open the file containing the measurements
        df = pd.read_csv(folder / sample_name, sep='\t', names=col_names)

        # create fft
        x, y, xf, yf = create_fft(df, x_name='Time', y_name=channel_name, sample_freq=20480.0, show_plot=False, window='kaiser', beta=3)
        # xf, yf = create_fft(df, x_name='Time', y_name=channel_name, sample_freq=20000.0, show_plot=False, window='kaiser', beta=3)

        # change sample name slightly to change '.' to '_' (personal preference)
        sample_name = sample_name.replace('.', '_')

        # append the time increments
        time_increment_seconds = unix_timestamp-start_time
        time_increment_days = time_increment_seconds /(60 * 60 * 24)
        
        # create new column for the current sample_name FFT
        if col_day_increment == False:
            dft[date_nice_format] = yf
        if col_day_increment == True:
            dft[str(time_increment_days)] = yf

        # create new dictionary key and values to store lable info
        labels_dict[sample_name] = [date_nice_format, sample_name, unix_timestamp, time_increment_seconds, time_increment_days]

    dft = dft.set_index(xf, drop=True) # index as frequency (Hz)
    return dft, labels_dict