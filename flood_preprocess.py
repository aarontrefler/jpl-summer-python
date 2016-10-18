# Aaron Trefler
# JPL
# Created: 2016-06-21
# Flood Observatory Preprocessing

############################################################
#SETUP
############################################################

import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio 

# define directories
dir_flood = '../../Raw Data/Flood Observatory/'

# import data
df_flood = pd.read_csv(dir_flood + 'MasterListrev.csv')

# flood observatory column descriptions
print 'List of FO Features:', list(df_flood.columns.values),'\n'

############################################################
#CLEAN
############################################################

# create dataframe with relevant columns
df_flood_sub = df_flood[[\
	'Register #', 'Began','Date Began', 'Ended', 'Duration in Days',\
	'Dead','Displaced', \
    'Main cause','Severity *','Affected sq km','Magnitude (M)**',
    'Country','Other', \
    'Centroid X', 'Centroid Y']]
# alter column names
#df_flood_sub = df_flood_sub.rename(columns = \
#    {'Detailed Locations (click on active links to access inundation extents)': \
#     'Detailed Locations'})

# create dataframe with records occuring 
# (1) after GRACE launch: 04/16/2002 
# (2) flood number 1907 began before GRACE launch, but ended after
# note final GRACE MASCON: 03/17/2016
df_flood_grace = df_flood_sub[(df_flood_sub['Register #'] > 1904) & \
    (df_flood_sub['Register #'] != 1907)]
print 'Flood Dataframe Shape:', df_flood_grace.shape,'\n'

# display NaN rows for 'Began' column
null_rows = df_flood_grace['Began'].isnull()
print 'Floods with no date:\n', df_flood_grace[['Register #', 'Began', 'Ended', 'Date Began']][null_rows],'\n'
# remove NaN rows
df_flood_grace = df_flood_grace[~null_rows]

############################################################
#SAVE
############################################################

df_flood_grace.to_csv('../Data/df_flood.csv')
