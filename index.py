
# Load libraries
import pandas

# Load dataset
uri = "data/ML_assignment 3_data.txt"
names = ['Date',
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
# 0: --- testing ---
# 506: ---- testing ----
dataset = pandas.read_csv(uri, sep=',', skiprows=[0, 506], header=None, names=names)

print(dataset.shape)

print(dataset)

# 530 lines - 3 lines = 527 lines

X = dataset.values
print(X[0])

# Split-out validation dataset

# Test options and evaluation metrics

# Training + Validation

