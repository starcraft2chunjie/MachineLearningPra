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
import math
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


#Module
class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

class yolo_like(nn.Module):
    def __init__(self):
        super(yolo_like, self).__init__()
        self.yolo = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding = 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 64, 3, padding = 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding = 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 128, 3, padding = 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128,256, 3, padding = 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 256, 3, padding = 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(256,512, 3, padding = 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 512, 3, padding = 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2)  # 5 * 5 * 512
        )
        self.flatten = Flatten()
        self.fc1 = nn.Linear(5 * 5 * 512, 1024)
        self.dropout = nn.Dropout()
        self.fc2 = nn.Linear(1024, 5 * 5 * (1 + 2))
    def forward(self, x):
        if torch.cuda.is_available():
            x = x.cuda()
        output = self.yolo(x)
        output = self.flatten(output)
        output = F.leaky_relu(self.fc1(output), 0.1)
        output = self.fc2(output)
        return output  #linear activation function

class loss(nn.Module):
    def __init__(self):
        super(loss, self).__init__()
        return
    def forward(self, pred, truth, S = 5):
        """args : pred : (batch, S * S * 3)
                  truth: (batch, n, [x, y])"""
        batch_size = pred.size(0)
        obj_num = truth.size(1)
        #ground truth
        #extract the coordinate prediction
        pred = pred.data
        truth = truth.data
        truth = truth.float()
        prediction = pred.contiguous().view(-1, S * S, 3)
        pred_coord = prediction[:, :, 1:3].contiguous()
        pred_conf = prediction[:, :, 0].contiguous()
        truth_posi = truth[:, :, 0:2]
        pred_coord1 = torch.zeros(batch_size, S * S, 2)
        # normalize to make it fall on 0-1
        truth_posi_x = ((truth[:, :, 0] + 14) % (80 / S)) * (S / 80) #batch_size * obj_num
        truth_posi_y = ((truth[:, :, 1] + 14) % (80 / S)) * (S / 80)
        label_ceil = ((truth[:, :, 0] + 14 - (truth[:, :, 0] + 14) % (80 / S)) / (80 / S)) * S + (truth[:, :, 1] + 14 - (truth[:, :, 1] + 14) % (80 / S)) / (80 / S)
        p_loss = torch.zeros(batch_size, S * S, 1).cuda()
        #Confidence = P * IOU 
        for i in range(batch_size):
            for j in range(obj_num):
                p_loss[i, int(label_ceil[i, j]), 0] = math.pow(prediction[i, int(label_ceil[i, j]), 0] - 1 * IOU(pred_coord[i, int(label_ceil[i, j]), 0], pred_coord[i, int(label_ceil[i, j]), 1], truth_posi_x[i, j], truth_posi_y[i, j]), 2)
        for i in range(batch_size):
            for j in range(S * S):
                for x in range(obj_num):
                    if j != label_ceil[i, x]:
                        p_loss[i, j, 0] =  0.5 * math.pow(prediction[i, j, 0] - 0.0, 2)
        pred_x = pred_coord[:, :, 0]
        pred_y = pred_coord[:, :, 1] # batch_size * SS
        dis_loss = torch.zeros(batch_size, S * S).cuda()
        for i in range(batch_size):
            for j in range(obj_num):
                dis_loss[i, j] = math.pow(pred_x[i, int(label_ceil[i, j])] - truth_posi_x[i, j], 2) + math.pow(pred_y[i, int(label_ceil[i, j])] - truth_posi_y[i, j], 2)
        dis_loss = 5 * dis_loss
        loss1 = torch.mean(dis_loss + p_loss, 1)
        loss2 = torch.mean(loss1, 0)
        return (Variable(loss2, requires_grad = True))

def IOU(pred_coord_x, pred_coord_y, truth_position_x, truth_position_y):
    #pred_coord is 2, x, y is relative to the grid
    #x is a num and y is a num relative to grid
    x1 = max((28 - 2 * abs(pred_coord_x * 16 - truth_position_x * 16)), 0)
    y1 = max((28 - 2 * abs(pred_coord_y * 16 - truth_position_y * 16)), 0)
    x2 = 28 + abs(pred_coord_x * 16 - truth_position_x * 16)
    y2 = 28 + abs(pred_coord_y * 16 - truth_position_y * 16)
    return (x1 * y1)/(x2 * y2)

model = yolo_like()
if torch.cuda.is_available():
    model.cuda()
optimizer = optim.Adam(model.parameters(), lr = 0.01 ,weight_decay=5e-4)
model.train()
train_loss = []
train_accu = []
i = 0
criterion = loss()
for epoch in range(10):
    for data, target in zip(batch_img, batch_location):
        data, target = Variable(data), Variable(target)
        if torch.cuda.is_available():
            data = data.cuda()
        output = model(data)
        ave_loss = criterion(output, target)
        optimizer.zero_grad()
        ave_loss.backward()
        train_loss.append(ave_loss.data[0])
        optimizer.step()
        if i % 80 == 0:
            print('Train Step: {}\tLoss: {:.3f}'.format(i, ave_loss.data[0]))
        i += 1


#performing non_maximum_suppression
def non_max_supp(output, S = 5, threshold = 0.1): #(S * S, 3)
    batch_size = output.size(0)
    output = output.data
    pro_output = output.contiguous().view(S * S, 3)
    for j in range(S * S):
        pro_output[j, 1] = pro_output[j, 1] * 16 + int((j+1) / 5) * 16
        pro_output[j, 2] = pro_output[j, 2] * 16 + (j - 5 * int((j+1) / 5)) * 16
    box_score = pro_output[:, 0]
    box_x = pro_output[:, 1]
    box_y = pro_output[:, 2]
    filter_mask = (box_score >= threshold)
    box_score = box_score[filter_mask]
    box_x = box_x[filter_mask]
    box_y = box_y[filter_mask]
    select_location = []
    for j in range(10):
        wait_location = []
        max_score_index = torch.max(box_score, 0)[1][0]
        max_score = torch.max(box_score, 0)[0][0]
        box_x_max = box_x[max_score_index]
        box_y_max = box_y[max_score_index]
        select_location.append([max_score, box_x_max, box_y_max])
        for i in range(len(box_score)):
            if IOU(box_x_max, box_y_max, box_x[i], box_y[i]) <= 0.4:
                wait_location.append(i)
        box_score = torch.take(box_score, torch.LongTensor(wait_location))
        box_x = torch.take(box_x, torch.LongTensor(wait_location))
        box_y = torch.take(box_y, torch.LongTensor(wait_location))
    return select_location

"""Then the operation is easy, just to put the image cropped to 28 * 28 * 1 to the Mnist model. But because the
   localization has some problems, I can't forward the operation."""











        




