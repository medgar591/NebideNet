import ramogen
from sklearn import linear_model
from sklearn import ensemble
from sklearn import neural_network
from sklearn import tree

trainingFile = "slrmodels.csv"
testingFile = "slrTestModels.csv"
createNewTrainingModels = False
createNewTestingModels = False

# Create new nets for training and testing
if createNewTrainingModels:
    print("Generating new training models")
    ramogen.genSLR(100,5000,trainingFile)
    ramogen.testSLR(trainingFile, "communitycrime/crimecommunity.csv", [100, 101], 0)
    print("Finished generating training models")
    
if createNewTestingModels:
    print("Generating new testing models")
    ramogen.genSLR(100,5000,testingFile)
    ramogen.testSLR(testingFile, "communitycrime/crimecommunity.csv", [100, 101], 0)
    print("Finished generating testing models")

# Read in files
files = open(trainingFile, "rt")
trainLinearNets = [list(map(float,item.split(","))) for item in files.read().splitlines()]
files.close()

trainLinearBiases = [row.pop(len(row)-1) for row in trainLinearNets]

files = open(testingFile, "rt")
testLinearNets = [list(map(float,item.split(","))) for item in files.read().splitlines()]
files.close()

testLinearBiases = [row.pop(len(row)-1) for row in testLinearNets]

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