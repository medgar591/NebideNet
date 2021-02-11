import torch
import torch.nn as nn
import torch.nn.functional as F
import ramogen

# Parameters
learning_rate = 0.01
criterion = nn.MSELoss()
trainingFile = "slrmodels.csv"
testingFile = "slrTestModels.csv"
createNewTrainingModels = False
createNewTestingModels = False

# Define Neural Net
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # TODO: Check to see if a better structure exists
        self.l1 = nn.Linear(100, 50)
        self.l2 = nn.Linear(50, 10)
        self.l3 = nn.Linear(10, 1)
    
    def forward(self, x):
        # TODO: Check here too
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x

net = Net()

# Create new nets for training and testing
if createNewTrainingModels:
    print("Generating new training models")
    ramogen.genSLR(100,1000,trainingFile)
    ramogen.testSLR(trainingFile, "communitycrime/crimecommunity.csv", [100, 101], 0)
    print("Finished generating training models")
    
if createNewTestingModels:
    print("Generating new testing models")
    ramogen.genSLR(100,1000,testingFile)
    ramogen.testSLR(testingFile, "communitycrime/crimecommunity.csv", [100, 101], 0)
    print("Finished generating testing models")

# Read in training file
files = open(trainingFile, "rt")
tempLinearNets = [list(map(float,item.split(","))) for item in files.read().splitlines()]
files.close()

tempLinearBiases = [row.pop(len(row)-1) for row in tempLinearNets]

linearNets = torch.tensor(tempLinearNets, dtype=torch.float)
linearBiases = torch.tensor(tempLinearBiases, dtype=torch.float)

#Training loop
print("Beginning Training")
for epoch in range(10):
    running_loss = 0.0
    for n in range (len(linearNets)):
        net.zero_grad()
        output = net(linearNets[n])
        loss = criterion(output, linearBiases[n])
        loss.backward()
        for f in net.parameters():
            f.data.sub_(f.grad.data * learning_rate)
        
        running_loss += loss.item()
        if n % 100 == 99:    # print every 100 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, n + 1, running_loss / 100))
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
tolerance = 0.005
correct = 0
total = 0
with torch.no_grad():
    for n in range (len(linearNets)):
        output = net(linearNets[n])
        total += 1
        if abs(linearBiases[n]-output) <= tolerance:
            correct += 1

print("Accuracy of the network on 1,000 test models: %d %%" % (correct/total*100))