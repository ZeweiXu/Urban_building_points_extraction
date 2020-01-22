Data and scripts for urban building extraction and change detection using multitemporal LiDAR data 
==============================================
Description:
The LiDAR data of Boston covers an area of 9 km2 and contains the central part of eastern Boston downtown and part of southern Boston consisting of industrial and residential areas. This area has an elevation ranging from 0 to 32m above sea level. The area is ideal for change detection research due to significant building changes from 2002 to 2014 captured by the multi-temporal LiDAR data. This area also has a complex urban building composition. There were multiple large construction projects carried out in this area from 2002 to 2014, including the Central Artery/Tunnel Project: https://en.wikipedia.org/wiki/Big_Dig) and the Boston Convention and Exhibition Center Project (https://en.wikipedia.org/wiki/Boston_Convention_and_Exhibition_Center). When the project was done and the Center has started to hold many events since June 2004, many buildings including hotels and business centers were constructed from the year 2004 to 2014.

Source: These data were acquired from MassGIS (Bureau of Geographic Information) (https://www.mass.gov/orgs/massgis-bureau-of-geographic-information)
Coding environment: Python2.7
size:3GB
Library dependencies: cuda7.0, pdal 1.2, sklearn 0.18.1, Pointnet 1.0 (https://github.com/charlesq34/pointnet)
Point density: 3 pts/m2
Horizontal accuracy: 2002: 0.5m  2014: 0.21m
Vertical accuracy: 2014: 0.15m 2014: 0.024

Classification steps:
1. Run data_preprocessing.py for LiDAR samples extraction with Pdal functions.
2. Train model by running train.py 
3. Run test.py to check output label file 

Folder structure:
----------------
  -Data
    - 2002.las, 2009.las, and 2014.las: Boston lidar data surveyed in 2002, 2009, and 2014.
    - totalarray: preprocessed patch-based Boston LiDAR for prediction. 
    - reference: folder that store the patch-based reference data from four different locations
    - predicted_labels.npy: predicted labels of each patch (0:non-building;1:building)
    - intermediate_data: intermediate dataset generated from preprocessing and model training and testing
  -Script
    - train.py: scripts to run point-based classification using PointSIFT for building classificaiton.
    - log20: folder that store model parameters and checkpoints
    - model.py: model structure and parameters
    - utils: folder used to put temoporary files



