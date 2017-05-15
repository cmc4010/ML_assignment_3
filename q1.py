
# small dataset
# easy version

# --- Load libraries ---
import pandas
import numpy as np
import matplotlib.pyplot as plt

# custom code: naive Bayes model
from nbdFunctions import binning, binQuery, basicNaiveBayesModel

# --- Load dataset ---

uri = "data/ML_assignment 3_basicdata.txt"
names = ['SS-IN', 'SED-IN', 'COND-IN', 'SS-OUT', 'SED-OUT', 'COND-OUT', 'STATUS']

dataset = pandas.read_csv(uri, sep=',', index_col = 0, names = names)

# print(dataset)

# CHECKPOINT #1 : dataset is correct!

# --- Understanding data + Notes ---

# print "shape of dataset: ", dataset.shape
# print(dataset)

# description = dataset.describe()
# print(description)

# dataset.hist()
# plt.show()

# 13 entries
# 6 descriptive feature
# 1 target feature

# PROBLEM: very little data
# SOLUTIONS: smoothing...

# PROBLEM: continuous features
# SOLUTIONS: normal distribution, exponential distribution

# list of headers
datasetColumns = dataset.columns.values
targetName = datasetColumns[-1]
# targetValueRange = ['ok','settler','solids']
targetValueRange = dataset[datasetColumns[-1]].unique()
# print targetValueRange
# featureRange = ['SS-IN', 'SED-IN', 'COND-IN', 'SS-OUT', 'SED-OUT', 'COND-OUT']
featureRange = datasetColumns[:-1]
# print featureRange

# custom boundaries created by eyeing the dataset histogram
boundaries = [
	[(256+964)/2], [(7 + 17)/2], [(1410 + 1814)/2],
	[(27+73)/2, (82+104)/2], [(3.5+1030)/2], [(43.1+1221)/2, (1575+1879)/2]
]

# bin data according to the custom boundaries
binnedData = binning( dataset, boundaries, featureRange )
myQuery = [222, 4.5, 1518, 74, 0.25, 1642]

# format the query for displaying
framedQuery = pandas.DataFrame([myQuery], index=["query"], columns=dataset.columns.values[:-1])

##### FINAL RESULTS #####

print "==== Query ====\n", framedQuery
print "==== Predictions ===="
print "Normal: ", basicNaiveBayesModel( dataset, myQuery, featureRange, targetName, targetValueRange, "normal" )
print "Exponential: ", basicNaiveBayesModel( dataset, myQuery, featureRange, targetName, targetValueRange, "exp" )
print "Binning+Smoothing: ", basicNaiveBayesModel( binnedData, myQuery, featureRange, targetName, targetValueRange, "discrete", "continuous", boundaries )
