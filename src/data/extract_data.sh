#!/bin/bash
PROJECT_DIR=$1

# move to raw data folder
cd $PROJECT_DIR/data/raw


# IMS data set
# Check if directory does not exist
if [ ! -f "IMS/IMS.7z" ]; then
    echo "Ensure IMS data downloaded"
else
    cd IMS
    7za e IMS.7z
    rar e 1st_test.rar
    unrar 2nd_test.rar
    unrar 3rd_test.rar
fi

# PRONOSTIA (FEMTO) data set
# Check if directory does not exist
if [ ! -f "FEMTO/FEMTOBearingDataSet.zip" ]; then
    echo "Ensure PRONOSTIA (FEMTO) data downloaded"
else
    cd FEMTO
    unzip FEMTOBearingDataSet.zip
    unzip Training_set.zip
    unzip Test_set.zip
fi
