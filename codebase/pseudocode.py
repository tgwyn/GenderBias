# Pseudocode for Project Iteration #1
# Tony Gwyn
# Jairen Gilmore
# Zhiwen He

# First step: import the various packages that we will need to make the determination
# about whether or not moving the business to the state of North Carolina
# will be a good idea

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Second step: import the dataset(s) using pandas
# Please note that the files are currently placeholders and not yet implemented

dataset1 = pd.read_csv('example1.csv')
dataset2 = pd.read_csv('example2.csv')

# Third step: clean the datasets, and get them ready to be processed by our algorithm
# As with the first step, this step is not yet fully implemented, and will be worked on
# in further project iterations

dataset1_cleaned = dataset1.clean()
dataset2_cleaned = dataset2.clean()

# Fourth step: pair down the datasets to the specific data that we want to work with.  This will 
# include numbers such as number of COVID-19 cases, hospitalizations, deaths, and recoveries. To 
# make sure the numbers are normalized per state, we will be taking a percentage value of these 
# figures rather than total number of cases.
# Again, this is currently pseudocode, and will be further implemented in P.I. #2 and beyond
# We will get the figures for NC, as well as the hypothetical state that the business is already
# located in, as well as 2 other states that the business could possibly be moved to.

NC_Figures_1 = dataset1_cleaned.loc[ "NC"], ["cases", "hospitalizations", "deaths", "recoveries",]
NC_Figures_2 = dataset2_cleaned.loc[ "NC"], ["cases", "hospitalizations", "deaths", "recoveries",]

TX_Figures_1 = dataset1_cleaned.loc[ "TX"], ["cases", "hospitalizations", "deaths", "recoveries",]
TX_Figures_2 = dataset2_cleaned.loc[ "TX"], ["cases", "hospitalizations", "deaths", "recoveries",]

SC_Figures_1 = dataset1_cleaned.loc[ "SC"], ["cases", "hospitalizations", "deaths", "recoveries",]
SC_Figures_2 = dataset2_cleaned.loc[ "SC"], ["cases", "hospitalizations", "deaths", "recoveries",]

GA_Figures_1 = dataset1_cleaned.loc[ "GA"], ["cases", "hospitalizations", "deaths", "recoveries",]
GA_Figures_2 = dataset2_cleaned.loc[ "GA"], ["cases", "hospitalizations", "deaths", "recoveries",]


# Fifth step: using established criteria given by the company leadership, determine whether or not
# moving the company's operation to NC is feasible, or a good idea when compared to data from other states
# The values currently used are placeholders, and will be further expounded upon in the next P.I.

maxPercentage = 0.05;

if CASESNC > (maxPercentage * NCPop)
	print('This research cannot recommend moving the base of operations to North Carolina, due to the amount of COVID-19 cases in the area')
	
else
	print('North Carolina is an acceptable location to move the base of operations to.')
	
# We will also run these figures for both the state the company is currently in, as well as the other candidates, 
# and give the overall best recommendation.


