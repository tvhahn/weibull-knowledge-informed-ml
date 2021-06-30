#!/bin/bash
PROJECT_DIR=$1
cd $PROJECT_DIR

mkdir -p $PROJECT_DIR/data/raw

# move to raw data folder
cd $PROJECT_DIR/data/raw

# IMS data set
# Check if directory does not exist
if [ ! -f IMS ]; then
    mkdir IMS
    cd IMS
    wget -4 https://ti.arc.nasa.gov/m/project/prognostic-repository/IMS.7z
    cd ..
fi


# PRONOSTIA (FEMTO) data set
# Check if directory does not exist
if [ ! -f FEMTO ]; then
    mkdir FEMTO
    cd FEMTO
    wget -4 https://ti.arc.nasa.gov/m/project/prognostic-repository/FEMTOBearingDataSet.zip
fi

