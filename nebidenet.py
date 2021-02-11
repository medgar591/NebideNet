import torch
import torch.nn as nn
import torch.nn.functional as F

learning_rate = 0.01
criterion = nn.MSELoss()
# loss = criterion(output, relatedBias)

files = open("slrmodels.csv", "rt")
tempLinearNets = [item.split(",") for item in files.read().splitlines()]
tempLinearNets[:] = [list(map(float, row)) for row in tempLinearNets]
files.close()

tempLinearBiases = [row.pop(len(row)-1) for row in tempLinearNets]

linearNets = torch.tensor(tempLinearNets, dtype=torch.float)
linearBiases = torch.tensor(tempLinearBiases, dtype=torch.float)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = nn.Linear(100, 50)
        self.l2 = nn.Linear(50, 10)
        self.l3 = nn.Linear(10, 1)
    
    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x

net = Net()
print(net)

#output = net(linearNets[1:10])
#print(output)


#Training loop
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