# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
import numpy as np



@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    # input file paths for FEMTO (PRONOSTIA) and IMS data sets
    folder_raw_data_train_femto = (
        input_filepath / "FEMTO/Training_set/Learning_set/"
    ) 

    folder_raw_data_test_femto = (
        input_filepath / "FEMTO/Test_set/Test_set/"
    )

    folder_raw_data_ims = (
        input_filepath / "IMS/"
    )

    # processed data file paths
    folder_processed_data_femto = (output_filepath / 'FEMTO/')
    folder_processed_data_ims = (output_filepath / 'IMS/')

    # make processed directories if not exist
    folder_processed_data_femto.mkdir(parents=True, exist_ok=True)
    folder_processed_data_ims.mkdir(parents=True, exist_ok=True)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    root_dir = Path(__file__).resolve().parents[2]


    main()
