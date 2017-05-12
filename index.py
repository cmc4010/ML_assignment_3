
# --- Load libraries ---
import pandas
from lookupTable import lookupGen

# --- Load dataset ---

uri = "data/ML_assignment 3_data.txt"
names = [ # 'Date',
'Q-E',
'ZN-E',
'PH-E',
'DBO-E',
'DQO-E',
'SS-E',
'SSV-E',
'SED-E',
'COND-E',
'PH-P',
'DBO-P',
'SS-P',
'SSV-P',
'SED-P',
'COND-P',
'PH-D',
'DBO-D',
'DQO-D',
'SS-D',
'SSV-D',
'SED-D',
'COND-D',
'PH-S',
'DBO-S',
'DQO-S',
'SS-S',
'SSV-S',
'SED-S',
'COND-S',
'RD-DBO-P',
'RD-SS-P',
'RD-SED-P',
'RD-DBO-S',
'RD-DQO-S',
'RD-DBO-G',
'RD-DQO-G',
'RD-SS-G',
'RD-SED-G']
# 1: --- testing ---
# 507: ---- testing ----
dataset = pandas.read_csv(uri, sep=',', skiprows=[0, 506], header=None, names=names)

print(dataset.shape)

print(dataset)

# LOOKUP TABLE

lookupTable = []

class1 = """D-1/3/90 to  D-12/3/90, D-16/3/90 to D-30/3/90, D-1/2/90 to D-19/2/90, D-21/2/90 to D-28/2/90,
    D-1/1/90 to D-26/1/90, D-29/1/90 to D-31/1/90, D-1/6/90 to D-4/6/90, D-6/6/90 to D-8/6/90,
    D-24/6/90, D-25/6/90, D-28/6/90, D-29/6/90, D-1/5/90 to D-6/5/90, D-8/5/90 to D-20/5/90,
    D-24/5/90, D-25/5/90, D-29/5/90, D-2/7/90, D-4/7/90 to D-8/7/90, D-12/7/90 to D-15/7/90, D-19/7/90, 
    D-23/7/90, D-26/7/90, D-4/9/90, D-5/9/90, D-23/9/90, D-28/9/90, D-30/9/90, D-17/8/90, D-21/8/90 to D-25/8/90,
    D-29/8/90, D-30/8/90, D-3/12/90, D-9/12/90, D-16/12/90 to D-20/12/90, D-23/12/90, D-24/12/90,
    D-27/12/90 to D-30/12/90,  D-6/11/90 to D-8/11/90, D-14/11/90, D-16/11/90, D-18/11/90,
    D-20/11/90, D-21/11/90, D-27/11/90, D-10/10/90, D-18/10/90, D-29/10/90, D-30/10/90,
    D-3/3/91 to D-6/3/91, D-10/3/91 to D-12/3/91, D-18/3/91, D-20/3/91, D-27/3/91, D-29/3/91,
    D-3/2/91, D-5/2/91, D-8/2/91, D-14/2/91, D-17/2/91, D-18/2/91, D-21/2/91 to D-24/2/91, 
    D-1/1/91, D-2/1/91, D-6/1/91, D-8/1/91, D-10/1/91 to D-20/1/91, D-25/1/91, D-2/5/91, D-3/5/91,
    D-7/5/91, D-14/5/91, D-15/5/91, D-17/5/91, D-19/5/91, D-21/5/91 to D-23/5/91, D-1/4/91 to D-3/4/91,
    D-5/4/91 to D-12/4/91, D-15/4/91 to D-21/4/91, D-23/4/91, D-1/7/91, D-3/7/91, D-4/7/91, D-7/7/91,
    D-10/7/91 to D-12/7/91, D-15/7/91, D-16/7/91, D-22/7/91 to D-25/7/91, D-28/7/91, D-30/7/91, D-31/7/91,
    D-2/6/91 to D-4/6/91, D-6/6/91, D-7/6/91, D-13/6/91, D-16/6/91 to D-21/6/91, D-25/6/91 to D-30/6/91,
    D-4/10/91, D-6/10/91, D-17/10/91 to D-30/10/91, D-1/8/91, D-2/8/91, D-27/8/91, D-29/8/91,   
    D-2/12/90, D-4/12/90, D-6/12/90, D-10/12/90 to D-14/12/90, D-21/12/90, D-26/12/90,
    D-15/11/90, D-22/11/90 to D-26/11/90, D-28/11/90 to D-30/11/90, D-19/10/90,
    D-13/3/91 to D-15/3/91, D-19/3/91, D-21/3/91, D-22/3/91, D-1/2/91, D-4/2/91,
    D-6/2/91, D-7/2/91, D-10/2/91 to  D-13/2/91, D-15/2/91, D-19/2/91,
    D-25/2/91 to D-28/2/91, D-3/1/91, D-4/1/91, D-7/1/91, D-21/1/91 to D-24/1/91,
    D-27/1/91 to D-31/1/91, D-6/5/91, D-4/4/91"""
classOne = lookupGen(class1)
lookupTable.append(classOne)
# print(len(classOne))

class2 = "D-13/3/90, D-14/3/90, D-15/3/90, D-17/7/91 to D-19/7/91"
classTwo = lookupGen(class2)
lookupTable.append(classTwo)
# print(len(classTwo))

class3 = """D-28/1/90, D-10/6/90 to D-22/6/90, D-26/6/90, D-27/6/90, D-7/5/90, D-21/5/90 to D-23/5/90,
    D-27/5/90, D-28/5/90, D-30/5/90, D-1/7/90,
    D-3/7/90, D-9/7/90 to D-11/7/90, D-16/7/90 to D-18/7/90, D-20/7/90, D-22/7/90, D-24/7/90, D-25/7/90,
    D-27/7/90 to D-31/7/90, D-2/9/90, D-3/9/90, D-6/9/90 to D-13/9/90, D-16/9/90 to D-21/9/90,
    D-24/9/90 to D-27/9/90, D-1/8/90 to D-7/8/90, D-16/8/90, D-28/8/90, D-31/8/90, D-7/12/90,
    D-2/11/90, D-5/11/90, D-9/11/90, D-12/11/90, D-13/11/90, D-1/10/90 to D-5/10/90, D-24/10/90,
    D-25/10/90, D-1/3/91, D-8/3/91, D-17/3/91, D-26/3/91, D-31/3/91, D-9/1/91, D-10/5/91, D-16/5/91,
    D-20/5/91, D-29/5/91, D-30/5/91, D-14/4/91, D-22/4/91, D-24/4/91, D-25/4/91, D-5/7/91, D-8/7/91,
    D-9/7/91, D-21/7/91, D-26/7/91, D-5/6/91, D-10/6/91, D-12/6/91, D-14/6/91, D-2/10/91, D-8/10/91,
    D-9/10/91, D-11/10/91, D-13/10/91, D-16/10/91"""
classThree = lookupGen(class3)
lookupTable.append(classThree)
# print(len(classThree))

class4 = "D-5/6/90, D-28/5/91, D-31/5/91, D-24/5/91"
classFour = lookupGen(class4)
lookupTable.append(classFour)
# print(len(classFour))

class5 = """D-8/8/90 to D-10/8/90, D-13/8/90, D-15/8/90, D-19/8/90, D-20/8/90, D-27/8/90, D-1/11/90, 
    D-4/11/90, D-11/11/90, D-19/11/90, D-7/10/90 to D-9/10/90, D-12/10/90 to D-17/10/90,
    D-21/10/90, D-23/10/90, D-26/10/90, D-28/10/90, D-7/3/91, D-24/3/91, D-25/3/91,
    D-1/5/91, D-5/5/91, D-8/5/91, D-9/5/91, D-12/5/91, D-13/5/91, D-26/5/91, D-27/5/91,
    D-26/4/91, D-28/4/91, D-29/4/91, D-2/7/91, D-14/7/91, D-29/7/91, D-9/6/91, D-24/6/91,
    D-1/10/91, D-3/10/91, D-5/10/91, D-12/10/91, D-15/10/91, D-4/8/91, D-9/8/91 to D-26/8/91,
    D-28/8/91, D-30/8/91"""
classFive = lookupGen(class5)
lookupTable.append(classFive)
# print(len(classFive))

class6 = "D-14/9/90, D-12/8/90, D-22/10/90"
classSix = lookupGen(class6)
lookupTable.append(classSix)
# print(len(classSix))

total = 0
for myClass in lookupTable:
	total += len(myClass)
	# print(len(myClass))
print(total)

# 530 lines - 3 lines = 527 lines

# X = dataset.values
# print(X[0])
# print(dataset.index[0])
# print(dataset.index[526])
# print(dataset.loc['D-1/3/90'])

# Classification (creating Y)
# for each value in dataset.index[i]
# check with the class lookup table
# generate Y[i]
Y = []

missed = []
flag = 0
for x in range(0, 527):
	for y in range(0, 6):
		if dataset.index[x] in lookupTable[y]:
			Y.append(y+1)
			flag = 1
			break
	if flag == 0:
		missed.append(x)
	flag = 0

# some entries are not classified

# print(missed)

# print(dataset.index[482])

# print(len(Y))

# --- Split-out validation dataset ---

# --- Test options and evaluation metrics ---

# --- Training + Validation ---

