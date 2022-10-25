#!/bin/bash
cd
cd scratch/

mkdir -p bearing_data_raw
cd bearing_data_raw
mkdir IMS
cd IMS
wget -4 https://phm-datasets.s3.amazonaws.com/NASA/4.+Bearings.zip
unzip -o 4.+Bearings.zip
mv '4. Bearings'/* .

cd ..
mkdir FEMTO
cd FEMTO
wget -4 https://phm-datasets.s3.amazonaws.com/NASA/10.+FEMTO+Bearing.zip
unzip -o 10.+FEMTO+Bearing.zip
mv '10. FEMTO Bearing'/* .
