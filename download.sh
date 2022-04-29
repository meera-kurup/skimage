#!/bin/bash

#this downloads the zip file that contains the data
kaggle datasets download -d gianmarco96/upmcfood101
# this unzips the zip file - you will get a directory named "data" containing the data
unzip upmcfood101.zip
# this cleans up the zip file, as we will no longer use it
rm upmcfood101.zip
# move to data folder
mv images data

echo downloaded data
