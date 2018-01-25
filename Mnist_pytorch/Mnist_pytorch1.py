import torch
import torch.nn as nn
import torch.nn.functional as F   
import torch.optim as optim 
from torchvision import datasets, transforms
from torch.autograd import Variable


class MnistModule(nn.Module):
    def __init__(self):
        super(MnistModule, self).__init__()
        #input is 28 * 28
        #padding = 2 for same padding
        self.conv1 = nn.Conv2d(1, 32, 5, padding = 2)
        #feature map size is 14 * 14 by pooling
        #paddding = 2 for same padding
        self.conv2 = nn.Conv2d(32, 64, 5, padding = 2)
        #filter num is used to be 2^n and not too small
        #feature map size is 7 * 7 by pooling
        self.fc1 = nn.Linear(64 * 7 * 7, 1024)
        self.fc2 = nn.Linear(1024, 10)
    def forward(self, x):
        if torch.cuda.is_available():
            x = x.cuda()
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # reshape Variable
        x = x.view(-1, 64 * 7 * 7) 
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training = self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim = 1)

model = MnistModule()
if torch.cuda.is_available():
    model.cuda()

batch_size = 50
train_loader = torch.utils.data.DataLoader(datasets.MNIST('inputs', train = True, download=True, transform=transforms.ToTensor()), batch_size=batch_size, shuffle = True)
test_loader = torch.utils.data.DataLoader(datasets.MNIST('inputs', train = False, transform=transforms.ToTensor()), batch_size=1000)

#Use Adam in order to combine the momentum and RMSprop
#Adam is said to be parted with lr = 0.001, so I use it, and find the result is satisfying
optimizer = optim.Adam(model.parameters(), lr = 0.001)
model.train()
train_loss = []
train_accu = []
i = 0
for epoch in range(10):
    for inputs, target in train_loader:
        #put the data and target into the Variable
        inputs, target = Variable(inputs), Variable(target)
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            target = target.cuda()
        optimizer.zero_grad()
        output = model(inputs)
        loss = F.nll_loss(output, target)
        loss.backward() #calc gradients
        train_loss.append(loss.data[0])
        optimizer.step() #update gradients
        prediction = output.data.max(1)[1] #first column has actual prob, get the index of it
        accuracy = prediction.eq(target.data).sum()/batch_size * 100
        train_accu.append(accuracy)
        if i % 10 == 0:
            print('Train Step: {}\tLoss: {:.3f}\tAccuracy: {:.3f}'.format(i, loss.data[0], accuracy))
        i += 1
model.eval()
correct = 0
for data, target in test_loader:
    data, target = Variable(data, volatile = True), Variable(target)
    output = model(data)
    if torch.cuda.is_available():
        inputs = inputs.cuda()
        target = target.cuda()
    prediction = output.data.max(1)[1]
    correct += prediction.eq(target.data).sum()
print('\nTest set: Accuracy: {:.2f}%'.format(100 * correct/len(test_loader.dataset)))
torch.save(model, 'Mnist1.pkl')
torch.save(model.state_dict(), 'Mnist1_params.pkl')

def restore_net():
    net2 = torch.load('Mnist1.pkl')







