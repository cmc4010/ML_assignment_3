
# small dataset
# easy version

# --- Load libraries ---
import math
import pandas
import numpy as np
import matplotlib.pyplot as plt

# --- Load dataset ---

uri = "data/ML_assignment 3_basicdata.txt"
names = ['SS-IN', 'SED-IN', 'COND-IN', 'SS-OUT', 'SED-OUT', 'COND-OUT', 'STATUS']

dataset = pandas.read_csv(uri, sep=',', index_col = 0, names = names)

# print(dataset)

# CHECKPOINT #1 : dataset is correct!

# --- Understanding data ---

# print "shape of dataset: ", dataset.shape
# print(dataset)

# description = dataset.describe()
# print(description)

# dataset.hist()
# plt.show()

targetValueRange = ['ok','settler','solids']
featureRange = ['SS-IN', 'SED-IN', 'COND-IN', 'SS-OUT', 'SED-OUT', 'COND-OUT']

# 13 entries
# 6 descriptive feature
# 1 target feature

# very little data
# SOLUTIONS: smoothing...

# continuous features
# SOLUTIONS: normal distribution, exponential distribution

# BINNING + SMOOTHING

# SS-IN: 3 bins... or 2
# -> boundary (2 bins): (256 + 964)/2
# SED-IN: 4 bins
# -> boundary (2 bins): (7 + 17)/2
# COND-IN: 2 big bins
# -> boundary (2 bins): (1410 + 1814)/2
# SS-OUT: 4 bins
# -> boundary (3 bins): (27+73)/2, (82+104)/2
# SED-OUT: 3 bins
# -> boundary (2 bins): (3.5+1030)/2
# COND-OUT: 3 bins
# -> boundary (3 bins): (43.1+1221)/2, (1575+1879)/2

# discretize into bin number 0, 1, (2)
# custom binning code

boundaries = [
	[(256+964)/2], [(7 + 17)/2], [(1410 + 1814)/2],
	[(27+73)/2, (82+104)/2], [(3.5+1030)/2], [(43.1+1221)/2, (1575+1879)/2]
]

# given dataset, boundaries, and feature range
# we can bin the dataset
def binning( dataset, boundaryList, featureList ):
	binnedDataset = dataset.values
	for idx, instance in enumerate(binnedDataset):
		# print instance
		for feature in range(0, len(featureList)):
			# 0~5 feature
			# perform for each feature
			updatedFlag = 0
			for bid, boundary in enumerate(boundaryList[feature]):
				# print boundary
				# print bid, instance[feature]
				if instance[feature] < boundary:
					instance[feature] = bid
					updatedFlag = 1
					break;
			# handle greater than boundary
			if updatedFlag == 0:
				instance[feature] = len(boundaryList[feature])
	# return 0
	return pandas.DataFrame(binnedDataset, index=dataset.index.values, columns=dataset.columns.values)

# --- Create a naive Bayes model ---

# the model is simple enough to code

# access the mean and std
# 0: ID
# 1: SS-IN
# 2: ...
# or you could just access with name as index
# print(dataset.mean(axis=0)[?])
# print(dataset.std(axis=0)[?])

# NORMAL distribution
# u: sample mean of the feature values
# o: standard deviation of the feature values
# we can find these values using dataset.describe()
def normalDist( value, u, o ):
	# u and o are feature specific
	# u = dataset.mean(axis=0)[feature]
	# o = dataset.std(axis=0)[feature]
	# calculate normal distribution
	exponent = -(value - u)**2 / (2 * o**2)
	result = ( 1/( o*(2*math.pi)**(1/2) )) * math.exp(exponent)
	return result

##### TESTING ZONE #####
# print normalDist( 222, "SS-IN" )
########################

# EXPONENTIAL distribution
# lambda: 1 / mean of the data
# we can find mean with dataset.describe()
def exponentialDist( value, lamb ):
	# lambda is feature specific
	# lamb = 1 / dataset.mean(axis=0)[feature]
	if value > 0:
		return lamb * math.exp(-lamb*value)
	else:
		return 0

##### TESTING ZONE #####
# print exponentialDist( 222, "SS-IN")
########################

def binQuery( query, boundaryList, featureList ):
	# value: query value
	# idx: index of which feature
	# boundaryList: list of boundaries to discretize the value
	binnedValues = []
	for feature in range(0, len(featureList)):
		updatedFlag = 0
		binnedValue = -1
		for bid, boundary in enumerate(boundaryList[feature]):
			# print boundary
			# print bid, instance[feature]
			if query[feature] < boundary:
				binnedValue = bid
				updatedFlag = 1
				break;
		# handle greater than boundary
		if updatedFlag == 0:
			binnedValue = len(boundaryList[feature])
		binnedValues.append(binnedValue)
	return binnedValues

def basicNaiveBayesModel( data, query, featureList, targetLevels, method, datatype = "discrete", boundaryList = None):
	# data: full dataset
	# query: query values in list format [q1, q2, q3, q4, q5, q6]
	# featureList: list of all descriptive features
	# targetLevels: list of possible target levels
	# method: type of distribution to use

	# multiply the probabilities together
	# IMPORTANT: the mean and std is calculated from subset of data
	# that has target that specific target value
	# print data
	results = []
	for target in targetLevels:
		# get subset based on target level
		targetSubset = data[data['STATUS']==target]
		targetProb = len(targetSubset)/float(len(data))
		# print targetSubset

		# BIN QUERY FOR CONTINUOUS DATA
		if method == "discrete" and datatype == "continuous":
			query = binQuery( query, boundaryList, featureList )
			# print query

		# PERFORM CALCULATIONS
		# query -- 6 entries
		# index 0 ~ 5
		total = 1
		for idx in range(0,len(featureList)):
			feature = featureList[idx]
			# print idx
			# NORMAL
			if method == "normal":
				u = targetSubset.mean(axis=0)[feature]
				o = targetSubset.std(axis=0)[feature]
				total *= normalDist( query[idx], u, o )
				# print normalDist( query[idx], u, o )
			# EXPONENTIAL
			elif method == "exp":
				lamb = 1 / targetSubset.mean(axis=0)[feature]
				total *= exponentialDist( query[idx], lamb)
				# print exponentialDist( query[idx], lamb)
			elif method == "discrete":
				# perform the most basic naive Bayes model operations
				# for each feature
				# 	check the relative frequency within target subset
				# PROBLEM: I have to handle "Empty dataframe"
				# SOLUTION: data smoothing

				# smoothing constant
				k = 1
				# subset of values equal to the query value
				eqVal = targetSubset[targetSubset[feature] == query[idx]]
				# count of the values equal to the query value
				countflt = len(eqVal)
				countft = len(targetSubset)
				domain = data[feature].unique()
				domainSize = len(domain)
				# print len(domain)
				# the probability of this query value within the subset
				normalProb = float(countflt)/len(targetSubset)
				# probability of this query with smoothing
				smoothProb = float(countflt + k)/(countft + k * domainSize)
				total *= smoothProb
			else:
				print("Error in choosing distribution")
		results.append(total*targetProb)

	# FIND THE TARGET LEVEL WITH HIGHEST PROBABILITY
	# print results
	myMax = max(results)
	indexMax = results.index(myMax)
	return targetLevels[indexMax]

##### FINAL RESULTS #####

binnedData = binning( dataset, boundaries, featureRange )
myQuery = [222, 4.5, 1518, 74, 0.25, 1642]

framedQuery = pandas.DataFrame([myQuery], index=["query"], columns=dataset.columns.values[:-1])

print "==== Query ====\n", framedQuery
print "==== Predictions ===="
print "Normal: ", basicNaiveBayesModel( dataset, myQuery, featureRange, targetValueRange, "normal" )
print "Exponential: ", basicNaiveBayesModel( dataset, myQuery, featureRange, targetValueRange, "exp" )
print "Binning+Smoothing: ", basicNaiveBayesModel( binnedData, myQuery, featureRange, targetValueRange, "discrete", "continuous", boundaries )
