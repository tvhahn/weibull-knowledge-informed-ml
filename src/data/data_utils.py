import numpy as np
import torch
import time
import datetime
import h5py
import fnmatch
import os
import csv

def get_min_max(x):
    """Get min/max value for an array
    
    Parameters
    ===========
    x : ndarray
        Signal or data set

    Returns
    ===========

    min_val : float
        Minimum value of the signal or dataset

    max_val : float
        Maximum value of the signal or dataset

    """

    # flatten the input array http://bit.ly/2MQuXZd
    flat_vector = np.concatenate(x).ravel()

    min_val = min(flat_vector)
    max_val = max(flat_vector)

    return min_val, max_val


def scaler(x, min_val, max_val, lower_norm_val=0, upper_norm_val=1):
    """Scale the signal between a min and max value
    
    Parameters
    ===========
    x : ndarray
        Signal that is being normalized

    max_val : int or float
        Maximum value of the signal or dataset

    min_val : int or float
        Minimum value of the signal or dataset

    lower_norm_val : int or float
        Lower value you want to normalize the data between (e.g. 0)

    upper_norm_val : int or float
        Upper value you want to normalize the data between (e.g. 1)

    Returns
    ===========
    x : ndarray
        Returns a new array that was been scaled between the upper_norm_val
        and lower_norm_val values

    """

    # https://codereview.stackexchange.com/questions/185785/scale-numpy-array-to-certain-range
    col, row = np.shape(x)
    for i in range(col):
        x[i] = np.interp(x[i], (min_val, max_val), (lower_norm_val, upper_norm_val))
    return x


def create_date_dict(folder):
    # instantiate the date dictionary that will
    # hold the date/time that each signal was recorded on
    # along with the file name

    date_dict = {}

    for i, file in enumerate(os.listdir(folder)):
        if fnmatch.fnmatch(file, f"acc*.csv"):

            # get the unix timestamp for when the file was modified (http://bit.ly/2RW5cYo)
            date_created = datetime.datetime.fromtimestamp(
                os.path.getmtime(folder / str(file))
            )

            # open each csv file, read first line, and extract times
            with open(folder / file, newline="") as f:
                csv_reader = csv.reader(f)
                csv_headings = next(csv_reader)

            # help with datetime: https://realpython.com/python-datetime/
            # convert "time" string to datetime object
            time_created = datetime.time(
                hour=int(float(csv_headings[0])),
                minute=int(float(csv_headings[1])),
                second=int(float(csv_headings[2])),
                microsecond=int(float(csv_headings[3])),
            )

            # combine date and time into a single datetime object
            combined_date = datetime.datetime.combine(date_created, time_created)
            unix_timestamp = combined_date.timestamp()
            date_nice_format = combined_date.strftime("%Y-%m-%d %H:%M:%S")

            date_dict[unix_timestamp] = [combined_date, date_nice_format, file]
    return date_dict


def create_x_y(df_spec, labels_dict, bucket_size=1000, print_shape=False):
    """Create x, y data set from the dataframe
    
    Parameters
    ===========
    df_spec : dataframe
        Dataframe that contains the data

    labels_dict : dict
        Dictionary that contains the labels

    bucket_size : int
        Number of samples to be used in each bucket

    print_shape : bool
        Whether or not to print the shape of the data


    Returns
    ===========
    x : ndarray
        Array of the x data
    
    y : ndarray
        Array of the y data. Each data point includes the run time since start (in days),
        the percentage life remaining, and the remaining useful life (in days)
    """

    samples = df_spec.shape[1]

    df_temp = df_spec.iloc[:10000]
    a = np.array(df_temp)  # make numpy array

    # get the max value for each bucket
    # https://stackoverflow.com/a/15956341/9214620
    x = np.max(a.reshape(-1, bucket_size, samples), axis=1).T

    temp_days = []
    for i in labels_dict.keys():
        temp_days.append(labels_dict[i][-1])

    temp_days = np.sort(np.array(temp_days))

    run_time = np.max(temp_days)

    # turn into percentage life left
    y = []
    for i in temp_days:
        y.append([i, i / run_time, run_time - i])

    y = np.array(y)

    # drop the last two values from the x and y, since they seem to be erroneous
    x = x[: len(x) - 2]
    y = y[: len(y) - 2]

    if print_shape == True:
        print("Shape of x:", np.shape(x))
        print("Shape of y:", np.shape(y))
        print("run-time-to-failure:", f"{run_time:.3f} days")
    return x, y


def load_train_test_ims(path):
    """Load the hdf5 files containing the train/val/test sets"""

    with h5py.File(path / "x_train.hdf5", "r") as f:
        x_train = f["x_train"][:]
    with h5py.File(path / "y_train.hdf5", "r") as f:
        y_train = f["y_train"][:]

    with h5py.File(path / "x_val.hdf5", "r") as f:
        x_val = f["x_val"][:]
    with h5py.File(path / "y_val.hdf5", "r") as f:
        y_val = f["y_val"][:]

    with h5py.File(path / "x_test.hdf5", "r") as f:
        x_test = f["x_test"][:]
    with h5py.File(path / "y_test.hdf5", "r") as f:
        y_test = f["y_test"][:]

    # second run
    with h5py.File(path / "x_train_2.hdf5", "r") as f:
        x_train_2 = f["x_train_2"][:]
    with h5py.File(path / "y_train_2.hdf5", "r") as f:
        y_train_2 = f["y_train_2"][:]

    # third run
    with h5py.File(path / "x_train_3.hdf5", "r") as f:
        x_train_3 = f["x_train_3"][:]
    with h5py.File(path / "y_train_3.hdf5", "r") as f:
        y_train_3 = f["y_train_3"][:]

    # convert to tensors
    x_train = torch.tensor(x_train).type(torch.float32)
    y_train = torch.tensor(y_train).type(torch.float32)

    x_val = torch.tensor(x_val).type(torch.float32)
    y_val = torch.tensor(y_val).type(torch.float32)

    x_test = torch.tensor(x_test).type(torch.float32)
    y_test = torch.tensor(y_test).type(torch.float32)

    x_train_2 = torch.tensor(x_train_2).type(torch.float32)
    y_train_2 = torch.tensor(y_train_2).type(torch.float32)

    x_train_3 = torch.tensor(x_train_3).type(torch.float32)
    y_train_3 = torch.tensor(y_train_3).type(torch.float32)

    return (
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
    )


def load_train_test_femto(path):
    """Load the hdf5 files containing the train/val/test sets"""

    with h5py.File(path / "x_train.hdf5", "r") as f:
        x_train = f["x_train"][:]
    with h5py.File(path / "y_train.hdf5", "r") as f:
        y_train = f["y_train"][:]

    with h5py.File(path / "x_val.hdf5", "r") as f:
        x_val = f["x_val"][:]
    with h5py.File(path / "y_val.hdf5", "r") as f:
        y_val = f["y_val"][:]

    with h5py.File(path / "x_test.hdf5", "r") as f:
        x_test = f["x_test"][:]
    with h5py.File(path / "y_test.hdf5", "r") as f:
        y_test = f["y_test"][:]

    #### TRAIN ####
    # Bearing1_1
    with h5py.File(path / "x_train1_1.hdf5", "r") as f:
        x_train1_1 = f["x_train1_1"][:]
    with h5py.File(path / "y_train1_1.hdf5", "r") as f:
        y_train1_1 = f["y_train1_1"][:]

    # Bearing2_1
    with h5py.File(path / "x_train2_1.hdf5", "r") as f:
        x_train2_1 = f["x_train2_1"][:]
    with h5py.File(path / "y_train2_1.hdf5", "r") as f:
        y_train2_1 = f["y_train2_1"][:]

    # Bearing3_1
    with h5py.File(path / "x_train3_1.hdf5", "r") as f:
        x_train3_1 = f["x_train3_1"][:]
    with h5py.File(path / "y_train3_1.hdf5", "r") as f:
        y_train3_1 = f["y_train3_1"][:]

    #### VAL ####
    # Bearing1_2
    with h5py.File(path / "x_val1_2.hdf5", "r") as f:
        x_val1_2 = f["x_val1_2"][:]
    with h5py.File(path / "y_val1_2.hdf5", "r") as f:
        y_val1_2 = f["y_val1_2"][:]

    # Bearing2_2
    with h5py.File(path / "x_val2_2.hdf5", "r") as f:
        x_val2_2 = f["x_val2_2"][:]
    with h5py.File(path / "y_val2_2.hdf5", "r") as f:
        y_val2_2 = f["y_val2_2"][:]

    # Bearing3_2
    with h5py.File(path / "x_val3_2.hdf5", "r") as f:
        x_val3_2 = f["x_val3_2"][:]
    with h5py.File(path / "y_val3_2.hdf5", "r") as f:
        y_val3_2 = f["y_val3_2"][:]

    #### TEST ####
    # Bearing1_3
    with h5py.File(path / "x_test1_3.hdf5", "r") as f:
        x_test1_3 = f["x_test1_3"][:]
    with h5py.File(path / "y_test1_3.hdf5", "r") as f:
        y_test1_3 = f["y_test1_3"][:]

    # Bearing2_3
    with h5py.File(path / "x_test2_3.hdf5", "r") as f:
        x_test2_3 = f["x_test2_3"][:]
    with h5py.File(path / "y_test2_3.hdf5", "r") as f:
        y_test2_3 = f["y_test2_3"][:]

    # Bearing3_3
    with h5py.File(path / "x_test3_3.hdf5", "r") as f:
        x_test3_3 = f["x_test3_3"][:]
    with h5py.File(path / "y_test3_3.hdf5", "r") as f:
        y_test3_3 = f["y_test3_3"][:]


    # convert to tensors
    x_train = torch.tensor(x_train).type(torch.float32)
    y_train = torch.tensor(y_train).type(torch.float32)

    x_val = torch.tensor(x_val).type(torch.float32)
    y_val = torch.tensor(y_val).type(torch.float32)

    x_test = torch.tensor(x_test).type(torch.float32)
    y_test = torch.tensor(y_test).type(torch.float32)

    x_train1_1 = torch.tensor(x_train1_1).type(torch.float32)
    y_train1_1 = torch.tensor(y_train1_1).type(torch.float32)

    x_train2_1 = torch.tensor(x_train2_1).type(torch.float32)
    y_train2_1 = torch.tensor(y_train2_1).type(torch.float32)

    x_train3_1 = torch.tensor(x_train3_1).type(torch.float32)
    y_train3_1 = torch.tensor(y_train3_1).type(torch.float32)

    x_val1_2 = torch.tensor(x_val1_2).type(torch.float32)
    y_val1_2 = torch.tensor(y_val1_2).type(torch.float32)

    x_val2_2 = torch.tensor(x_val2_2).type(torch.float32)
    y_val2_2 = torch.tensor(y_val2_2).type(torch.float32)

    x_val3_2 = torch.tensor(x_val3_2).type(torch.float32)
    y_val3_2 = torch.tensor(y_val3_2).type(torch.float32)

    x_test1_3 = torch.tensor(x_test1_3).type(torch.float32)
    y_test1_3 = torch.tensor(y_test1_3).type(torch.float32)

    x_test2_3 = torch.tensor(x_test2_3).type(torch.float32)
    y_test2_3 = torch.tensor(y_test2_3).type(torch.float32)

    x_test3_3 = torch.tensor(x_test3_3).type(torch.float32)
    y_test3_3 = torch.tensor(y_test3_3).type(torch.float32)


    return (
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
    )