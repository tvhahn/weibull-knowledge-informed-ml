import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal, fftpack
import seaborn as sns
import time
import datetime
from typing import Tuple


def create_fft(
    df, y_name="b2_ch3", sample_freq=20480.0, window="hamming", beta=8.0
) -> Tuple[np.array, np.array]:
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
    x = np.arange(0, df.shape[0], dtype="float64") / (sample_freq)

    # parameters for plot
    T = 1.0 / sample_freq  # sample spacing
    N = len(y)  # number of sample points

    # do some preprocessing of the current signal
    y_detrend = y - np.mean(y)
    y_detrend = signal.detrend(y_detrend, type="constant")  # detrended signal

    if window == "hamming":
        y_detrend *= np.hamming(
            N
        )  # apply a hamming window. Why? https://dsp.stackexchange.com/a/11323
    else:
        y_detrend *= np.kaiser(len(y_detrend), beta)

    # FFT on time domain signal
    yf = fftpack.rfft(y_detrend)
    yf = 2.0 / N * np.abs(yf[: int(N / 2.0)])
    xf = np.linspace(0.0, 1.0 / (2.0 * T), N // 2) / 2

    return x, y, xf, yf


def build_spectrogram_df_ims(
    folder,
    date_list,
    channel_name="b3_ch5",
    start_time="2003.10.22.12.06.24",
    col_day_increment=False,
    col_names=[
        "b1_ch1",
        "b1_ch2",
        "b2_ch3",
        "b2_ch4",
        "b3_ch5",
        "b3_ch6",
        "b4_ch7",
        "b4_ch8",
    ],
):
    """function that builds the spectrogram data for the IMS data"""

    # convert start_time to unix timestamp
    start_time = time.mktime(
        datetime.datetime.strptime(start_time, "%Y.%m.%d.%H.%M.%S").timetuple()
    )

    # instantiate dataframe for the spectrogram
    dft = pd.DataFrame()

    # dictionary to store any labels
    labels_dict = {}

    # iterate through each date that samples were taken
    # date_list should be sorted from earliest to latest
    for i, sample_name in enumerate(date_list):
        # convert sample_name to unix timestamp
        unix_timestamp = time.mktime(
            datetime.datetime.strptime(sample_name, "%Y.%m.%d.%H.%M.%S").timetuple()
        )
        date_nice_format = datetime.datetime.fromtimestamp(unix_timestamp).strftime(
            "%Y-%m-%d %H:%M:%S"
        )  # reformat date

        # open the file containing the measurements
        df = pd.read_csv(folder / sample_name, sep="\t", names=col_names)

        # create fft
        x, y, xf, yf = create_fft(
            df,
            y_name=channel_name,
            sample_freq=20480.0,
            window="kaiser",
            beta=3,
        )

        # change sample name slightly to change '.' to '_' (personal preference)
        sample_name = sample_name.replace(".", "_")

        # append the time increments
        time_increment_seconds = unix_timestamp - start_time
        time_increment_days = time_increment_seconds / (60 * 60 * 24)

        # create new column for the current sample_name FFT
        if col_day_increment == False:
            dft[date_nice_format] = yf
        if col_day_increment == True:
            dft[str(time_increment_days)] = yf

        # create new dictionary key and values to store lable info
        labels_dict[sample_name] = [
            date_nice_format,
            sample_name,
            unix_timestamp,
            time_increment_seconds,
            time_increment_days,
        ]

    dft = dft.set_index(xf, drop=True)  # index as frequency (Hz)
    return dft, labels_dict


def build_spectrogram_df_femto(
    folder,
    date_dict,
    channel_name="acc_horz",
    col_day_increment=False,
    col_names=["hr", "min", "sec", "micro_sec", "acc_horz", "acc_vert"],
):
    """function that builds the spectrogram data for the PRONOSTIA (FEMTO) data"""

    # date_time list
    date_list = sorted(list(date_dict.keys()))
    start_time = date_list[0]  # get the star time

    # instantiate dataframe for the spectrogram
    dft = pd.DataFrame()

    # dictionary to store any labels
    labels_dict = {}

    # iterate through each date that samples were taken
    # date_list should be sorted from earliest to latest
    for i, unix_timestamp in enumerate(date_list):
        # convert sample_name to unix timestamp
        date_nice_format = date_dict[unix_timestamp][1]

        # open the file containing the measurements
        df = pd.read_csv(folder / date_dict[unix_timestamp][2], names=col_names)

        # create fft
        x, y, xf, yf = create_fft(
            df,
            y_name=channel_name,
            sample_freq=25600.0,
            window="kaiser",
            beta=3,
        )

        # append the time increments
        time_increment_seconds = unix_timestamp - start_time
        time_increment_days = time_increment_seconds / (60 * 60 * 24)

        # create new column for the current sample_name FFT
        if col_day_increment == False:
            dft[date_nice_format] = yf
        if col_day_increment == True:
            dft[str(time_increment_days)] = yf

        # create new dictionary key and values to store lable info
        labels_dict[unix_timestamp] = [
            date_nice_format,
            unix_timestamp,
            time_increment_seconds,
            time_increment_days,
        ]

    dft = dft.set_index(xf, drop=True)  # index as frequency (Hz)
    return dft, labels_dict
