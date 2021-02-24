import ramogen
from sklearn import linear_model
from sklearn import ensemble
from sklearn import neural_network
from sklearn import tree
from sklearn import svm

trainingFile = "slrmodels.csv"
testingFile = "slrTestModels.csv"
createNewTrainingModels = False
createNewTestingModels = False
numModels = 500 # Number of models to use for testing
classify = True # Used to determine if models should work as regression or classification
fmodel = 1 # Used to determine which model of fairness to use, 0 = parity, 1 = ratio

# Create new nets for training and testing
if createNewTrainingModels:
    print("Generating new training models")
    ramogen.genSLR(100,numModels,trainingFile)
    ramogen.testSLR(trainingFile, "communitycrime/crimecommunity.csv", [100, 101], 0, fmodel, classify)
    print("Finished generating training models")
    
if createNewTestingModels:
    print("Generating new testing models")
    ramogen.genSLR(100,numModels,testingFile)
    ramogen.testSLR(testingFile, "communitycrime/crimecommunity.csv", [100, 101], 0, fmodel, classify)
    print("Finished generating testing models")

# Read in files
files = open(trainingFile, "rt")
trainLinearNets = [list(map(float,item.split(","))) for item in files.read().splitlines()]
files.close()

trainLinearBiases = [row.pop(len(row)-1) for row in trainLinearNets]
if classify:
    trainLinearBiases[:] = [bool(item) for item in trainLinearBiases]

files = open(testingFile, "rt")
testLinearNets = [list(map(float,item.split(","))) for item in files.read().splitlines()]
files.close()

testLinearBiases = [row.pop(len(row)-1) for row in testLinearNets]
if classify:
    testLinearBiases[:] = [bool(item) for item in testLinearBiases] 

# Default model results:
# LinearRegression = 20.8%
# Ridge = 20.8%
# Lasso = 40.96%
# ElasticNet = 40.96%
# Lars = 20.8%
# LassoLars = 40.96%
# OrthagonalMatchingPursuit = 17.24%
# BayesianRidge = 20.9%
# ARDRegression = 19.08%
# TweedieRegressor = 28%
# SGDRegressor = 20.22%
# Perceptron = NA - doesn't like the standard inputs
# PassiveAggressiveRegressor = 15.48%
# RANSACRegressor = 5.14% !!
# TheilSenRegressor = 20.78% - super slow
# HuberRegressor = 21.68% - related to Ridge?

# AdaBoostRegressor = 13.06%
# Bagging Regerssor = -0.5% base?
# ExtraTreesRegressor = 24.44%
# GradientBoostingRegressor = 20.34%
# IsolationForest = 0 - Must have used it wrong
# RandomForestRegressor = 21.76%

# MLPRegressor = 8.28%

# DecisionTreeRegressor = 20.58%
# ExtraTreeRegressor = 20.16%
if not classify:
    model = linear_model.Lasso()
    print("Starting fitting")
    model.fit(trainLinearNets, trainLinearBiases)
    print("Done fitting")

    predictions = model.predict(testLinearNets)
    tolerance = 0.025
    total = 0
    correct = 0
    for n in range (len(testLinearNets)):
        total += 1
        if abs(testLinearBiases[n]-predictions[n]) <= tolerance:
            correct += 1
    print("Accuracy of the network on after fitting: %f %%" % (100.0*correct/total))
else:
    model = svm.LinearSVC()
    print("Starting fitting")
    model.fit(trainLinearNets, trainLinearBiases)
    print("Done fitting")

    predictions = model.predict(testLinearNets)
    total = 0
    correct = 0
    for n in range (len(testLinearNets)):
        total += 1
        if (bool(testLinearBiases[n]) == bool(predictions[n])):
            correct += 1
    print("Accuracy of the network on after fitting: %f %%" % (100.0*correct/total))