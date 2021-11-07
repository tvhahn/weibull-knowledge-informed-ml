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
    unrar x 1st_test.rar
    unrar x 2nd_test.rar
    unrar x 3rd_test.rar
    mkdir 3rd_test
    cd 4th_test
    mv txt/* ../3rd_test/
    cd $PROJECT_DIR/data/raw
fi

# move to raw data folder
cd $PROJECT_DIR/data/raw

# PRONOSTIA (FEMTO) data set
# Check if directory does not exist
if [ ! -f "FEMTO/FEMTOBearingDataSet.zip" ]; then
    echo "Ensure PRONOSTIA (FEMTO) data downloaded"
else
    cd FEMTO
    unzip -o FEMTOBearingDataSet.zip
    unzip -o Training_set.zip
    unzip -o Test_set.zip
fi
