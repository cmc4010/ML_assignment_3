# Machine Learning Assignment 3 Overview

**NOTE:** The contents of the "overview" section is taken from the assignment description.

## Background

Predictive data analytics models are often used as tools for process quality control and fault detection.

## Objective

To create a naive Bayes model to monitor a waste water treatment plant.

## Understanding Data

14 days  
6 descriptive features  
1 target feature

descriptive features (6, all continuous):
- SS-IN, SS-OUT : amount of solids
- SED-IN, SED-OUT : amount of sediments
- COND-IN, COND-OUT : electrical conductivity

target feature (3 levels):
1. ok, everything is working correctly
2. settler, there is a problem with the plant settler equipment
3. solids, there is a problem with the amount of solids going through the plant

## Task Breakdown

a. Create a naive Bayes model that uses probability density functions to model the descriptive features in this dataset (smoothing, normally distributed, exponential, binning, ... or more).

b. What prediction will the naive Bayes model return for the following query?

>SS-IN = 222, SED-IN = 4.5, COND-IN = 1,518, SS-OUT = 74, SED-OUT = 0.25, COND-OUT = 1,642

**NOTE:** "Testing...determine D-1/4/90 to D-30/4/90"

## What to hand in?

1. Source code
2. Report (incl. results, environment, library and language, code explanation and how to use it)

# Implementation

## Environment and Library

Language: Python  
Environment: macOS Sierra V10.12.4 (Macbook Pro, 13-inch, 2016)

**NOTE:** There is a line of code in lookupTable.py that did not run on my Windows machine. The code is only tested on macOS.

Library Descriptions  
Pandas: Powerful Python data analysis toolkit  
SciKit-Learn: Python modules for machine learning and data mining  
Matplotlib: Plotting library for Python  
Numpy: N-dimensional array for numerical computation

## Naive Bayes Model

Probability Density Functions  
1.	Normal  
2.	Exponential

Built-in Models (SciKit-Learn)  
1.	Gaussian  
2.	Multinomial  
3.	Bernoulli

## Problems

### Dataset Parsing/Classification
The dataset was given without their respective classes in the csv file. I chose to create a lookup table for each class and then map our dataset to its respective class.

### Unclassified Data
The training data size is 504 entries. 4 entries were not classified so I chose to remove them completely from the dataset. That means the remaining training data size is 500.

### Missing Values
A possible solution is to simply exclude them from the product of evidence events which will work for discrete features.

The solution I went with was to *impute* the missing values by the mean value of the feature. This may not be a good solution for small datasets. However, with 500 instances of training data, I think it is a viable solution.

### No Match
>Smoothing involves taking some of the probability mass from the assignments with probability greater than average and spreading it across the probabilities that are below average, or even equal to zero.

Laplace smoothing, also called additive smoothing, was used in conjunction with binning. Binning discretizes the continuous features. To solve the issue produced by “no matches”, we perform smoothing on the binned dataset.

The binning boundaries were chosen by eyeing the dataset histogram as well as the data table. Midpoints between values that differ by a lot were chosen as binning boundaries. By examining the dataset, you can see that specific ‘STATUS’s have certain feature values that are a lot higher or a lot lower than other “STATUS’s which means that binning is a very logical solution despite the limited dataset.

## Code Overview

**lookupTable.py**
>contains code that parses a string to create an array of values that will be used as a lookup table in **q2.py**

**nbdFunctions.py**
>contains code for the custom naïve Bayes model 

**q1.py**
>contains code that trains and predicts using basic dataset

**q2.py**
>contains code that trains and predicts using large dataset

### BASIC DATA OVERVIEW

**Highlight:** Examine dataset to choose appropriate binning boundaries. Once binning is complete, perform Laplace smoothing.

### LARGE DATA OVERVIEW

1.	Generate lookup table (already given the information about which instance belongs to which class, but we must pair the instances with their respective class ourselves)
2.	Classify each instance of data with the lookup table
3.	Clean out data that are not classified
4.	Impute data so that we can do model training
5.	Train model
6.	Predict class for testing data

## Results

Source code: **p1.py**  
Models created for small dataset:  
1.	Normal  
2.	Exponential  
3.	Binning + Smoothing

Source code: **p2.py**  
Models created for large dataset:  
1.	Normal  
2.	Exponential  
3.	Gaussian  
4.	Multinomial  
5.	Bernoulli

## Conclusion

As you can see, most of the work done for the assignment is data parsing/handling. The models that uses probability density functions are simpler to code so it makes a pretty good coding assignment for learning Python. However, as a machine learning assignment, I don’t think this much emphasis should be placed on handling data (maybe).
