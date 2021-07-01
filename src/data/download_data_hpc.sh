#!/bin/bash
cd
cd scratch/

mkdir -p bearing_data_raw
cd bearing_data_raw
mkdir IMS
cd IMS
wget -4 https://ti.arc.nasa.gov/m/project/prognostic-repository/IMS.7z

cd ..
mkdir FEMTO
cd FEMTO
wget -4 https://ti.arc.nasa.gov/m/project/prognostic-repository/FEMTOBearingDataSet.zip

