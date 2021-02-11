import torch
import torch.nn as nn
import torch.nn.functional as F
import ramogen

# Parameters
learning_rate = 0.004 #ideal value seems to lie in [0.003, 0.01]
repetitions = 5 #Hits diminishing returns after 4
criterion = nn.MSELoss()
trainingFile = "slrmodels.csv"
testingFile = "slrTestModels.csv"
createNewTrainingModels = False
createNewTestingModels = False

# Basic set of linear layers in sequence for the net
net = nn.Sequential(
    nn.Linear(100,50),
    nn.Linear(50, 50),
    nn.Linear(50, 1)
)

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

# Read in training file
files = open(trainingFile, "rt")
tempLinearNets = [list(map(float,item.split(","))) for item in files.read().splitlines()]
files.close()

tempLinearBiases = [row.pop(len(row)-1) for row in tempLinearNets]

linearNets = torch.tensor(tempLinearNets, dtype=torch.float)
linearBiases = torch.tensor(tempLinearBiases, dtype=torch.float)

tolerance = 0.025
correct = 0
total = 0
with torch.no_grad():
    for n in range (len(linearNets)):
        output = net(linearNets[n])
        total += 1
        if abs(linearBiases[n]-output) <= tolerance:
            correct += 1

print("Accuracy of the network on before training: %f %%" % (100.0*correct/total))

#Training loop
print("Beginning Training")
for epoch in range(repetitions):
    running_loss = 0.0
    for n in range (0, len(linearNets), 5):
        output = net(linearNets[n:(n+5)])#.squeeze() # TODO: find out why squeezing this to remove the warning reduces accuracy by 1/3
        net.zero_grad()
        loss = criterion(output, linearBiases[n:(n+5)])
        loss.backward()
        with torch.no_grad():
            for param in net.parameters():
                param -= param.grad * learning_rate
        #for f in net.parameters():
        #    f.data.sub_(f.grad.data * learning_rate)
        
        running_loss += loss.item()
        if n % 1000 == 995:    # print every 1000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, n + 5, running_loss / 1000))
            running_loss = 0.0

print('Finished Training')

# Reading in testing file
files = open(testingFile, "rt")
tempLinearNets = [list(map(float,item.split(","))) for item in files.read().splitlines()]
files.close()

tempLinearBiases = [row.pop(len(row)-1) for row in tempLinearNets]

linearNets = torch.tensor(tempLinearNets, dtype=torch.float)
linearBiases = torch.tensor(tempLinearBiases, dtype=torch.float)

# Testing loop
tolerance = 0.025
correct = 0
total = 0
with torch.no_grad():
    for n in range (len(linearNets)):
        output = net(linearNets[n])
        total += 1
        if abs(linearBiases[n]-output) <= tolerance:
            correct += 1

print("Accuracy of the network on 5,000 test models: %f %%" % (100.0*correct/total))