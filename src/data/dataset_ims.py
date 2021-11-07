import numpy as np
from pathlib import Path
import os
import h5py


# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils import shuffle

# import custom functions and classes
from data_utils import (
    get_min_max,
    scaler,
    create_x_y,
    create_date_dict,
)

from src.features.build_features import build_spectrogram_df_ims


###################
# Create Data Set
###################


def create_ims_dataset(
    folder_raw_data, folder_processed_data, bucket_size=500, random_state_int=694
):
    """Create the IMS processed data, with appropriate train/val/test sets

    Parameters
    ===========
    folder_raw_data : pathlib object
        Location of raw data, both train and test, likely ./data/raw/IMS/

    folder_processed_data : pathlib object
        Location to store processed data (.h5py files). Likely ./data/processed/IMS/

    bucket_size : int
        The number of data points (from FFT spectrum) to include in each bin, or bucket.
        For example, if we want 20 bins on the IMS data set, we should set the bucket
        size to 500. The average, or max value, is taken from each bucket to make the
        final vector of size 20 for each time step (vector of 20 fed into neural network)

    random_state_int : int
        Number to reproduce the data split

    Returns
    ===========
    A bunch of .h5py files, in the folder_processed_data, for each of the respective
    train/validation/testing sets.

    """

    print("IMS data prep start.")

    #### TRAIN ####
    # 2nd RUN
    # For x_train, y_train
    # Bearing 1, outer race, (b1_ch1) failed
    folder_2nd = folder_raw_data / "2nd_test"
    date_list2 = sorted(os.listdir(folder_2nd))
    col_names = ["b1_ch1", "b2_ch2", "b3_ch3", "b4_ch4"]
    df_spec2, labels_dict2 = build_spectrogram_df_ims(
        folder_2nd,
        date_list2,
        channel_name="b1_ch1",
        start_time=date_list2[0],
        col_names=col_names,
    )
    print("created spectrogram for b1_ch1")
    ####

    # 3rd RUN
    # For x_train, y_train
    # Bearing 1, outer race, (b3_ch3) failed
    folder_3rd = folder_raw_data / "3rd_test"
    date_list3 = sorted(os.listdir(folder_3rd))
    col_names = ["b1_ch1", "b2_ch2", "b3_ch3", "b4_ch4"]
    df_spec3, labels_dict3 = build_spectrogram_df_ims(
        folder_3rd,
        date_list3,
        channel_name="b3_ch3",
        start_time=date_list3[0],
        col_names=col_names,
    )
    print("created spectrogram for b3_ch3")
    ####

    #### VAL ####
    # 1st RUN
    # For x_val, y_val
    # Bearing 3, inner race, (b3_ch6) failed <--- SHOULD CHANGE TO HORIZONTAL BEARING
    folder_1st = folder_raw_data / "1st_test"
    date_list1 = sorted(os.listdir(folder_1st))
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
    df_spec1_3, labels_dict1_3 = build_spectrogram_df_ims(
        folder_1st,
        date_list1,
        channel_name="b3_ch5",
        start_time=date_list1[0],
        col_names=col_names,
    )
    print("created spectrogram for b3_ch6")

    #### TEST ####
    # 1st RUN
    # For x_test, y_test
    # Bearing 4, rolling element, (b4_ch8) failed <--- SHOULD CHANGE TO HORIZONTAL BEARING
    folder_1st = folder_raw_data / "1st_test"
    date_list1 = sorted(os.listdir(folder_1st))
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
    df_spec1_4, labels_dict1_4 = build_spectrogram_df_ims(
        folder_1st,
        date_list1,
        channel_name="b4_ch7",
        start_time=date_list1[0],
        col_names=col_names,
    )
    print("created spectrogram for b4_ch8")
    ####

    # create the x-y for the train sets
    x2, y2 = create_x_y(df_spec2, labels_dict2, bucket_size, print_shape=False)
    x3, y3 = create_x_y(df_spec3, labels_dict3, bucket_size, print_shape=False)
    t2 = np.max(y2[:, 0])  # get the run-time in days
    t3 = np.max(y3[:, 0])

    # bold printout https://stackoverflow.com/a/17303428/9214620
    print(
        f"\033[1mTest 2\033[0m run-time: {t2:.3f} days \t\t\033[1mTest 3\033[0m run-time: {t3:.3f} days"
    )

    # calculate the weibull properties
    beta = 2.0  # shape parameter
    r = 2  # number of failed bearings
    i = 8  # number of bearings

    t_array = np.append([t2] * 4, [t3] * 4)  # build a time array of t

    # characteristic life
    eta = (np.sum((t_array ** beta) / r)) ** (1 / beta)
    eta_beta_r = np.array([eta, beta, r])

    print("eta:", eta)

    #######
    # Create x, y train/val/test
    #######

    x_train = np.append(x2, x3, 0)
    y_train = np.append(y2, y3, 0)

    x_val, y_val = create_x_y(df_spec1_3, labels_dict1_3, bucket_size)
    x_val = x_val[1:]
    y_val = y_val[1:]

    x_test, y_test = create_x_y(df_spec1_4, labels_dict1_4, bucket_size)
    x_test = x_test[1:]
    y_test = y_test[1:]

    # shuffle
    x_train, y_train = shuffle(x_train, y_train, random_state=random_state_int)
    x_val, y_val = shuffle(x_val, y_val, random_state=random_state_int)
    x_test, y_test = shuffle(x_test, y_test, random_state=random_state_int)

    # scale
    min_val, max_val = get_min_max(x_train)
    x_train = scaler(x_train, min_val, max_val)
    x_val = scaler(x_val, min_val, max_val)
    x_test = scaler(x_test, min_val, max_val)

    # create data set for the second and third runs (which are combined into x_train)
    # so that we can easily trend the results
    # second run
    x_train_2 = x2[1:]
    x_train_2 = scaler(x_train_2, min_val, max_val)  # scale
    y_train_2 = y2[1:]

    # third run
    x_train_3 = x3[1:]
    x_train_3 = scaler(x_train_3, min_val, max_val)  # scale
    y_train_3 = y3[1:]

    with h5py.File(folder_processed_data / "x_train.hdf5", "w") as f:
        dset = f.create_dataset("x_train", data=x_train)
    with h5py.File(folder_processed_data / "y_train.hdf5", "w") as f:
        dset = f.create_dataset("y_train", data=y_train)

    with h5py.File(folder_processed_data / "x_val.hdf5", "w") as f:
        dset = f.create_dataset("x_val", data=x_val)
    with h5py.File(folder_processed_data / "y_val.hdf5", "w") as f:
        dset = f.create_dataset("y_val", data=y_val)

    with h5py.File(folder_processed_data / "x_test.hdf5", "w") as f:
        dset = f.create_dataset("x_test", data=x_test)
    with h5py.File(folder_processed_data / "y_test.hdf5", "w") as f:
        dset = f.create_dataset("y_test", data=y_test)

    # save eta/beta
    with h5py.File(folder_processed_data / "eta_beta_r.hdf5", "w") as f:
        dset = f.create_dataset("eta_beta_r", data=eta_beta_r)

    # save t_array
    with h5py.File(folder_processed_data / "t_array.hdf5", "w") as f:
        dset = f.create_dataset("t_array", data=t_array)

    # second run
    with h5py.File(folder_processed_data / "x_train_2.hdf5", "w") as f:
        dset = f.create_dataset("x_train_2", data=x_train_2)
    with h5py.File(folder_processed_data / "y_train_2.hdf5", "w") as f:
        dset = f.create_dataset("y_train_2", data=y_train_2)

    # third run
    with h5py.File(folder_processed_data / "x_train_3.hdf5", "w") as f:
        dset = f.create_dataset("x_train_3", data=x_train_3)
    with h5py.File(folder_processed_data / "y_train_3.hdf5", "w") as f:
        dset = f.create_dataset("y_train_3", data=y_train_3)
