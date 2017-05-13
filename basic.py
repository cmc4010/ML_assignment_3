
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

print(dataset.shape)

description = dataset.describe()

# print(description)

X = dataset.values

X_train = X[:,0:6]
Y_train = X[:,6:]

# print(X_train)
# print(Y_train)

targetValueRange = ['ok','settler','solids']
featureRange = ['SS-IN', 'SED-IN', 'COND-IN', 'SS-OUT', 'SED-OUT', 'COND-OUT']

# dataset.hist()
# plt.show()

# 13 entries
# 6 descriptive feature
# 1 target feature

# very little data
# SOLUTIONS: smoothing...

# continuous features
# SOLUTIONS: normal distribution, exponential distribution

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

# print normalDist( 222, "SS-IN" )

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

# print exponentialDist( 222, "SS-IN")

def basicNaiveBayesModel( query, featureList, targetLevels, method ):
	# query: query values in list format [q1, q2, q3, q4, q5, q6]
	# featureList: list of all descriptive features
	# targetLevels: list of possible target levels
	# method: type of distribution to use

	# multiply the probabilties together
	# IMPORTANT: the mean and std is calculated from subset of data
	# that has target that specific target value
	results = []
	for target in targetLevels:
		# get subset based on target level
		targetSubset = dataset[dataset['STATUS']==target]
		targetProb = len(targetSubset)/float(len(dataset))

		# PERFORM CALCULATIONS
		# query -- 6 entries
		# index 0 ~ 5
		total = 1
		for idx in range(0,len(featureList)):
			feature = featureList[idx]
			# NORMAL
			if method == "normal":
				u = targetSubset.mean(axis=0)[feature]
				o = targetSubset.std(axis=0)[feature]
				total *= normalDist( query[idx], u, o )
			# EXPONENTIAL
			elif method == "exp":
				lamb = 1 / dataset.mean(axis=0)[feature]
				total *= exponentialDist( query[idx], lamb)
			else:
				print("Error in choosing distribution")
		results.append(total*targetProb)

	# FIND THE TARGET LEVEL WITH HIGHEST PROBABILITY
	print results
	myMax = max(results)
	indexMax = results.index(myMax)
	return targetLevels[indexMax]

myQuery = [222, 4.5, 1518, 74, 0.25, 1642]
print basicNaiveBayesModel( myQuery, featureRange, targetValueRange, "exp" )
