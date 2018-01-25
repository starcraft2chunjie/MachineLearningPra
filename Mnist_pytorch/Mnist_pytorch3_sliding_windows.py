#yolo algorthim
#9 * 9 will be ok for two number and 19 * 19 for multiply number
#evaluate method: intersection over union IoU = overlap/union
#Non-max-Suppression
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import scipy.io
import scipy.misc
import numpy as np
import pandas as pd
import PIL
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.autograd import Variable
from torchvision import transforms
import pickle

#load the data
pkl_file = open('two_numbers_dataset_6k.pkl', 'rb')
raw_datasets = pickle.load(pkl_file, encoding = 'bytes')
img = []
label = []
location = []

batch_size = 20

for i in range(len(raw_datasets)):
    img1 = transforms.ToTensor()(raw_datasets[i][0].reshape(80, 80, 1))
    img.append(img1)
    label.append([raw_datasets[i][1][0][0], raw_datasets[i][1][1][0]])
    location.append([raw_datasets[i][1][0][1:], raw_datasets[i][1][1][1:]])
train_len = int(2 * len(raw_datasets)/3)
train_img = img[:train_len]
test_img = img[train_len:len(raw_datasets)]
tr_list_label = label[:train_len]
te_list_label = label[train_len:len(raw_datasets)]
tr_list_location = location[:train_len]
te_list_location = location[train_len:len(raw_datasets)]
train_label = []
test_label = []
train_location = []
test_location = []
for i in range(train_len):
    train_label.append(torch.from_numpy(np.array(tr_list_label[i])))
    train_location.append(torch.from_numpy(np.array(tr_list_location[i])))
for i in range(len(raw_datasets) - train_len):
    test_label.append(torch.from_numpy(np.array(te_list_label[i])))
    test_location.append(torch.from_numpy(np.array(te_list_location[i])))
batch_img = []
batch_label = []
batch_location = []
for i in range(int(train_len/batch_size)):
    batch_img.append(torch.stack(train_img[batch_size * i : batch_size * (i+1)], 0))
    batch_label.append(torch.stack(train_label[batch_size * i : batch_size * (i+1)], 0))
    batch_location.append(torch.stack(train_location[batch_size * i : batch_size * (i + 1)], 0))
# for one small image is 28 * 28, So we need to resize the initial image to: 14 * 14, stride is 4
# module one: use the p as the variable to update, generate 14 * 14 windows to detect
# module two: use the coordinate (x, y) of four bounding boxes to update to localization


"""method one"""
class localization_module(nn.Module):
    def __init__(self):
        super(localization_module, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 5, padding = 2) #80 * 80
        self.conv2 = nn.Conv2d(16, 32, 5, padding = 2) #40 * 40
        self.conv3 = nn.Conv2d(32, 64, 5) #16 * 16 
        self.conv4 = nn.Conv2d(64, 128, 3) #14 * 14
        self.conv5 = nn.Conv2d(128, 1, 1) #14 * 14
    def forward(self, x):
        if torch.cuda.is_available():
            x = x.cuda()
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return F.sigmoid(x) 
"""method two"""
class regression_local_module(nn.Module):
    def __init__(self):
        super(regression_local_module, self).__init__()
        #extract the feature
        self.conv1 = nn.Conv2d(1, 16, 5, padding = 2) #80 * 80
        self.conv2 = nn.Conv2d(16, 32, 5, padding = 2) #40 * 40
        self.conv3 = nn.Conv2d(32, 64, 5, padding = 2)#20 * 20
        self.conv4 = nn.Conv2d(64, 128, 5, padding = 2)#10 * 10
        self.fc1 = nn.Linear(128 * 5 * 5, 1024)
        self.fc2 = nn.Linear(1024, 8)
    def forward(self, x):
        if torch.cuda.is_available():
            x = x.cuda()
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        x = F.max_pool2d(F.relu(self.conv4(x)), 2)
        x = x.view(-1, 128 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training = self.training)
        x = self.fc2(x)
        return x
#method one 
"""model = localization_module()
if torch.cuda.is_available():
    model.cuda()
optimizer = optim.Adam(model.parameters(), lr = 0.001)
train_loss = []
train_accu = []
for epoch in range(10):
    for data, target in zip(batch_img, batch_location):
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        target = torch.squeeze(target)
        customize_loss = loss()
        loss1 = customize_loss(output, target, batch_size)[0]
        p = customize_loss(output, target, batch_size)[1]
        loss1.backward()
        train_loss.append(loss1.data[0])
        optimizer.step()
        if i % 80 == 0:
            print('Train Step: {}\tLoss: {:.3f}'.format(i, loss1.data[0]))
        i += 1
"""
#method two
def split(grounding_truth):
    truth1 = torch.index_select(grounding_truth, 1, torch.LongTensor([0])) #batch_size * 1 * 4
    truth2 = torch.index_select(grounding_truth, 1, torch.LongTensor([1])) #batch_size * 1 * 4
    truth1 = torch.index_select(truth1, 2, torch.LongTensor([0, 1])) #batch_size * 1 * 2
    truth2 = torch.index_select(truth2, 2, torch.LongTensor([0, 1])) #batch_size * 1 * 2
    return (truth1, truth2)
model = regression_local_module()
if torch.cuda.is_available():
    model.cuda()
optimizer = optim.Adam(model.parameters(), lr = 0.01)
train_loss = []
train_accu = []
func_loss = torch.nn.MSELoss()
for epoch in range(10):
    for data, target in zip(batch_img, batch_location):
        target1, target2 = split(target)
        data, target1, target2 = Variable(data), Variable(target1), Variable(target2)
        if torch.cuda.is_available():
            data = data.cuda()
            target1 = target1.cuda()
            target2 = target2.cuda()
        output = model(data)
        output1, output2, output3, output4 = torch.chunk(output, 4, dim = 1)
        target1, target2 = target1.float(), target2.float()
        loss1 = torch.min(func_loss(output1, target1), func_loss(output1, target2))
        loss2 = torch.min(func_loss(output2, target1), func_loss(output2, target2))
        loss3 = torch.min(func_loss(output3, target1), func_loss(output3, target2))
        loss4 = torch.min(func_loss(output4, target1), func_loss(output4, target2))
        loss = (loss1 + loss2 + loss3 + loss4)/4
        optimizer.zero_grad()
        loss.backward()
        train_loss.append(loss.data[0])
        optimizer.step()
        if i % 80 == 0:
            print('Train Step: {}\tLoss: {:.3f}'.format(i, loss.data[0]))
        i += 1

#The two method has some problems, and then I find another method. So I quit in advance.    

