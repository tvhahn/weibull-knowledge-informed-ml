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

from scipy.stats import pointbiserialr


"""
Gather all the result csv's, combine them together, and then append the test scores.

Filter out poorly performing models and save the final result csv's in the models/final folder.

Save the top performing models also in the models/final folder.
"""

# set the folder location of the temporary results
root_dir = Path.cwd()
print(root_dir)
temp_dir = root_dir / 'models/interim/results_csv_ims'

# use multi-processing to load all the CSVs into one file
# https://stackoverflow.com/a/36590187
# wrap your csv importer in a function that can be mapped
def read_csv(filename):
    'converts a filename to a pandas dataframe'
    return pd.read_csv(filename)


def main(folder_path):

    # get a list of file names
    files = os.listdir(folder_path)
    file_list = [temp_dir / filename for filename in files if filename.endswith('.csv')]

    # set up your pool
    with Pool(processes=7) as pool: # or whatever your hardware can support

        # have your pool map the file names to dataframes
        df_list = pool.map(read_csv, file_list)

        # reduce the list of dataframes to a single dataframe
        combined_df = pd.concat(df_list, ignore_index=True)
        
        return combined_df
        

if __name__ == '__main__':
    df = main(temp_dir)
    
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
ADD_TEST_RESULTS = True # True or False

if ADD_TEST_RESULTS:
    folder_path = root_dir / 'data/processed/IMS/'

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
    ) = load_train_test_ims(folder_path)
    
    
#     (
#         x_train,
#         y_train,
#         x_val,
#         y_val,
#         x_test,
#         y_test,
#         x_train1_1,
#         y_train1_1,
#         x_train2_1,
#         y_train2_1,
#         x_train3_1,
#         y_train3_1,
#         x_val1_2,
#         y_val1_2,
#         x_val2_2,
#         y_val2_2,
#         x_val3_2,
#         y_val3_2,
#         x_test1_3,
#         y_test1_3,
#         x_test2_3,
#         y_test2_3,
#         x_test3_3,
#         y_test3_3,
#     ) = load_train_test_femto(folder_path)


    # load beta, eta for Weibull CDF
    with h5py.File(folder_path / "eta_beta_r.hdf5", "r") as f:
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
    
    model_folder = root_dir / 'models/interim/checkpoints_ims/'
    print(model_folder)

    # append test results onto results dataframe
    df_results = test_metrics_to_results_df(model_folder, df, x_test, y_test)
    
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