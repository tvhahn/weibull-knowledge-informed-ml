import numpy as np
import pandas as pd
import seaborn as sns
import os
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from multiprocessing import Pool

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from src.models.model import Net

from src.data.data_utils import load_train_test_ims, load_train_test_femto
from src.models.utils import test, calc_r2_avg, model_metrics_test, test_metrics_to_results_df
from src.models.loss import RMSELoss, RMSLELoss
import src.models.model
# from visualizations import plot_trained_model_results
from sklearn.metrics import r2_score

import h5py
from pathlib import Path

import fnmatch
import os
import shutil
import sys

from scipy.stats import pointbiserialr


"""
Gather all the result csv's, combine them together, and then append the test scores.

Filter out poorly performing models and save the final result csv's in the models/final folder.

Save the top performing models also in the models/final folder.
"""

#####
# Set Parameters
SAVE_ENTIRE_CSV = False # if you want to save the entire CSV, before filtering
ADD_TEST_RESULTS = True # if you want to append the test results

#####

# check if "scratch" path exists in the home directory
# if it does, assume we are on HPC
scratch_path = Path.home() / 'scratch'

DATASET_TYPE = str(sys.argv[1])  # 'ims' or 'femto'

# set the default directories
if scratch_path.exists():
    # set important folder locations
    print('Assume on HPC')
    root_dir = Path.cwd()
    print('#### Root dir:', root_dir)

    if DATASET_TYPE == "ims":
        folder_data = Path.cwd() / "data/processed/IMS/"
    else:
        folder_data = Path.cwd() / "data/processed/FEMTO/"

    folder_results = Path(scratch_path / f"weibull_results/results_csv_{DATASET_TYPE}")
    folder_checkpoints = Path(scratch_path / f"weibull_results/checkpoints_{DATASET_TYPE}")
    folder_learning_curves = Path(scratch_path / f"weibull_results/learning_curves_{DATASET_TYPE}")

else:
    # set important folder locations
    print('Assume on local compute')
    root_dir = Path.cwd()
    print('#### Root dir:', root_dir)

    # data folder
    if DATASET_TYPE == "ims":
        folder_data = root_dir / "data/processed/IMS/"
        print("load IMS data", folder_data)
    else:
        folder_data = root_dir / "data/processed/FEMTO/"
        print("load FEMTO data", folder_data)

    folder_results = root_dir / f"models/interim/results_csv_{DATASET_TYPE}"
    folder_checkpoints = root_dir / f"models/interim/checkpoints_{DATASET_TYPE}"
    folder_learning_curves = root_dir / f"models/interim/learning_curves_{DATASET_TYPE}"


# use multi-processing to load all the CSVs into one file
# https://stackoverflow.com/a/36590187
# wrap your csv importer in a function that can be mapped
def read_csv(filename):
    'converts a filename to a pandas dataframe'
    return pd.read_csv(filename)


def main(folder_path):

    # get a list of file names
    files = os.listdir(folder_path)
    file_list = [folder_results / filename for filename in files if filename.endswith('.csv')]

    # set up your pool
    with Pool(processes=7) as pool: # or whatever your hardware can support

        # have your pool map the file names to dataframes
        df_list = pool.map(read_csv, file_list)

        # reduce the list of dataframes to a single dataframe
        combined_df = pd.concat(df_list, ignore_index=True)
        
        return combined_df
        

if __name__ == '__main__':
    df = main(folder_results)
    
# drop first column
try:
    df = df.drop(columns='Unnamed: 0')
except:
    pass

# export combined dataframe
# csv sav_name
csv_save_name = 'combined_results_2021.04.05_1.csv'

# add a unique identifier for each model architecture
df['date_time_seed'] = df['date_time'].astype(str)+'_'+df['rnd_seed_input'].astype(str)

# get name that model checkpoint was saved under
df['model_checkpoint_name'] = df['date_time'].astype(str)+'_'+df['loss_func'] +'_'+df['rnd_seed_input'].astype(str)+'.pt'

# move 'date_time_seed' to front
# df = df[[list(df).pop()] + list(df)[:-1]]

df.to_csv(csv_save_name, index=False)
print('Final df shape:',df.shape)

#### append test results to df ####
if ADD_TEST_RESULTS:

    if DATASET_TYPE == "ims":
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
        ) = load_train_test_ims(folder_data)
        
    else:
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
        ) = load_train_test_femto(folder_data)


    # load beta, eta for Weibull CDF
    with h5py.File(folder_data / "eta_beta_r.hdf5", "r") as f:
        eta_beta_r = f["eta_beta_r"][:]

    ETA = eta_beta_r[0]
    BETA = eta_beta_r[1]

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
    
    # append test results onto results dataframe
    df_results = test_metrics_to_results_df(folder_checkpoints, df, x_test, y_test)
    
    standard_losses = ['mse', 'rmse', 'rmsle']

    # apply 0 or 1 for weibull, and for each unique loss func
    for index, value in df_results['loss_func'].items():
        if value in standard_losses:
            df_results.loc[index, 'weibull_loss'] = 0
        else:
            df_results.loc[index, 'weibull_loss'] = 1

    # convert to 'weibull_loss' column to integer
    df_results['weibull_loss'] = df_results['weibull_loss'].astype(int)


    loss_func_list = df_results['loss_func'].unique()

    for index, value in df_results['loss_func'].items():
        for loss_func in loss_func_list:
            df_results.loc[index, value] = 1

    df_results[loss_func_list] = df_results[loss_func_list].fillna(0, downcast='infer')
    
    
    df_results.to_csv('combined_results_2021.04.05_1_with_test.csv', index=False)
    
print(df.head())