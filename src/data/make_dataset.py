# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
import numpy as np
from dotenv import find_dotenv, load_dotenv

#!#!#!#!# Helper Functions #!#!#!#!#

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


##########################################################################


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
