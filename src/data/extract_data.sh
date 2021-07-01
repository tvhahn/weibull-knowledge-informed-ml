#!/bin/bash
PROJECT_DIR=$1

mkdir -p $PROJECT_DIR/data/raw/IMS
mkdir -p $PROJECT_DIR/data/raw/FEMTO

# move to the scratch folder to download data
cd
cd scratch/

echo $PWD

if [ ! -d "bearing_data_temp" ]; then
    echo "Ensure data downloaded"
else
    cd bearing_data_temp
    cd IMS
    7za e IMS.7z
    unrar x 1st_test.rar
    unrar x 2nd_test.rar
    unrar x 3rd_test.rar
    mkdir 3rd_test
    cd 4th_test
    mv txt/* ../3rd_test/
    cp -r 1st_test $PROJECT_DIR/data/raw/IMS/1st_test
    cp -r 2nd_test $PROJECT_DIR/data/raw/IMS/2nd_test
    cp -r 3rd_test $PROJECT_DIR/data/raw/IMS/3rd_test

    cd
    cd scratch/bearing_data_temp/FEMTO
    unzip FEMTOBearingDataSet.zip
    unzip Training_set.zip
    unzip Test_set.zip
    cp -r Learning_set $PROJECT_DIR/data/raw/FEMTO/Learning_set
    cp -r Test_set $PROJECT_DIR/data/raw/FEMTO/Test_set

fi


# IMS data set
# Check if directory does not exist
# if [ ! -f "IMS/IMS.7z" ]; then
#     echo "Ensure IMS data downloaded"
# else
#     cd IMS
#     7za e IMS.7z
#     unrar x 1st_test.rar
#     unrar x 2nd_test.rar
#     unrar x 3rd_test.rar
#     mkdir 3rd_test
#     cd 4th_test
#     mv txt/* ../3rd_test/
#     cd $PROJECT_DIR/data/raw
# fi

# move to raw data folder
# cd $PROJECT_DIR/data/raw

# # PRONOSTIA (FEMTO) data set
# # Check if directory does not exist
# if [ ! -f "FEMTO/FEMTOBearingDataSet.zip" ]; then
#     echo "Ensure PRONOSTIA (FEMTO) data downloaded"
# else
#     cd FEMTO
#     unzip FEMTOBearingDataSet.zip
#     unzip Training_set.zip
#     unzip Test_set.zip
# fi
