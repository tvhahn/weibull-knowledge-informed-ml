#!/bin/bash
PROJECT_DIR=$1

cd $PROJECT_DIR

python $PROJECT_DIR/src/data/download_data_local.py --path_data_folder $PROJECT_DIR/data

