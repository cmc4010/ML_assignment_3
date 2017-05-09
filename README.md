
# ML_assignment_3

## Background

Predictive data analytics models are often used as tools for process quality control and fault detection.

## Objective

To create a naive Bayes model to monitor a waste water treatment plant.

## Understanding Data

14 days
6 descriptive features
1 target feature

descriptive features (6, all continuous):
1. SS-IN, SS-OUT : amount of solids
2. SED-IN, SED-OUT : amount of sediments
3. COND-IN, COND-OUT : electrical conductivity

target feature (3 levels):
1. ok, everything is working correctly
2. settler, there is a problem with the plant settler equipment
3. solids, there is a problem with the amount of solids going through the plant

## Task Breakdown

a. Create a naive Bayes model that uses probability density functions to model the descriptive features in this dataset (smoothing, normally distributed, exponential, binning, ... or more).

b. What prediction will the naive Bayes model return for the following query?

	...
	...

**NOTE:** "Testing...determine D-1/4/90 to D-30/4/90"

## What to hand in?

1. Source code
2. Report (incl. results, environment, library and language, code explanation and how to use it)