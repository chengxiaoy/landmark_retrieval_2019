#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 10:00:24 2018
Paper: Siamese Neural Networks for One-shot Image Recognition
links: https://www.cnblogs.com/denny402/p/7520063.html
"""
import torch
from torch.autograd import Variable
import os
import random
import linecache
import numpy as np
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import PIL.ImageOps
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("cuda is {} available".format(torch.cuda.is_available()))


class Config():
    root = '/Users/tezign/PycharmProjects/CV_learn/kaggle/landmark_retrieval/'
    train_txt = '/Users/tezign/PycharmProjects/CV_learn/kaggle/landmark_retrieval/' \
                'annotations_landmarks/annotation_clean_train.txt'
    test_txt = '/Users/tezign/PycharmProjects/CV_learn/kaggle/landmark_retrieval/' \
               'annotations_landmarks/annotation_clean_val.txt'
    train_batch_size = 32
    train_number_epochs = 100


# Helper functions
def imshow(img, text=None, should_save=False):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic', fontweight='bold',
                 bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def show_plot(iteration, loss):
    plt.plot(iteration, loss)
    plt.show()


def convert(train=True):
    if (train):
        train_f = open(Config.train_txt, 'w')
        test_f = open(Config.test_txt, 'w')

        for i in range(1, 41):
            for j in range(1, 11):
                rand_int = random.randint(1, 4)
                img_path = Config.data_path + 's' + str(i) + '/' + str(j) + '.pgm'
                if rand_int == 1:
                    test_f.write(img_path + ' ' + str(i) + '\n')
                else:
                    train_f.write(img_path + ' ' + str(i) + '\n')
        train_f.close()
        test_f.close()


convert(True)


#
# ready the dataset, Not use ImageFolder as the author did
class MyDataset(Dataset):

    def __init__(self, txt, transform=None, target_transform=None, should_invert=False):

        self.transform = transform
        self.target_transform = target_transform
        self.should_invert = should_invert
        self.txt = txt
        self.data = self.build_dict()
        self.labels = list(self.data.keys())

    def build_dict(self):
        """
        将图片地址 加载到内存当中  因为landmark的category太多 不太适合每次读file
        :return:
        """
        f = open(self.txt, 'r')
        clean_data = f.readlines()
        f.close()

        pic_dict = {}
        for line in clean_data:
            url = line.split(" ")[0]
            label = line.split(" ")[1]
            if pic_dict.__contains__(label):
                pic_dict[label].append(url)
            else:
                pic_dict[label] = [url]
        return pic_dict

    def __getitem__(self, index):

        label1 = random.choice(self.labels)
        should_get_same_class = random.randint(0, 1)
        if should_get_same_class:
            img1_path, img2_path = random.sample(self.data[label1], 2)
        else:
            img1_path = random.choice(self.data[label1])
            while True:
                label2 = random.choice(self.labels)
                if label1 != label2:
                    img2_path = random.choice(self.data[label2])
                    break

        img0 = Image.open(img1_path)
        img1 = Image.open(img2_path)
        # 转换成灰度图
        img0 = img0.convert("L")
        img1 = img1.convert("L")

        if self.should_invert:
            img0 = PIL.ImageOps.invert(img0)
            img1 = PIL.ImageOps.invert(img1)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        if should_get_same_class:
            return img0, img1, torch.from_numpy(np.array([0], dtype=np.float32))
        else:
            return img0, img1, torch.from_numpy(np.array([1], dtype=np.float32))


    def __len__(self):
        fh = open(self.txt, 'r')
        num = len(fh.readlines())
        fh.close()
        return num


# Visualising some of the data
"""
train_data=MyDataset(txt = Config.txt_root, transform=transforms.ToTensor(), 
                     transform=transforms.Compose([transforms.Scale((100,100)),
                               transforms.ToTensor()], should_invert=False))
train_loader = DataLoader(dataset=train_data, batch_size=8, shuffle=True)
#it = iter(train_loader)
p1, p2, label = it.next()
example_batch = it.next()
concatenated = torch.cat((example_batch[0],example_batch[1]),0)
imshow(torchvision.utils.make_grid(concatenated))
print(example_batch[2].numpy())
"""

# Neural Net Definition, Standard CNNs
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()

        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, 4, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),
            nn.Dropout2d(p=.2),

            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
            nn.Dropout2d(p=.2),

            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
            nn.Dropout2d(p=.2),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(8 * 100 * 100, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 5)
        )

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2


# Custom Contrastive Loss
class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive


# Training
train_data = MyDataset(txt=Config.train_txt, transform=transforms.Compose(
    [transforms.Resize((100, 100)), transforms.ToTensor()]), should_invert=False)
train_dataloader = DataLoader(dataset=train_data, shuffle=True, num_workers=4, batch_size=Config.train_batch_size)

# net = SiameseNetwork()
net = SiameseNetwork().to(device)
criterion = ContrastiveLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0005)

counter = []
loss_history = []
iteration_number = 0


for epoch in range(0, Config.train_number_epochs):
    for i, data in enumerate(train_dataloader, 0):
        img0, img1, label = data

        img0, img1, label = img0.to(device), img1.to(device), label.to(device)
        # img0, img1, label = Variable(img0).cuda(), Variable(img1).cuda(), Variable(label).cuda()
        output1, output2 = net(img0, img1)
        # 先将梯度归零
        optimizer.zero_grad()
        loss_contrastive = criterion(output1, output2, label)
        loss_contrastive.backward()
        optimizer.step()

        if i % 10 == 0:
            print("Epoch:{},  Current loss {}\n".format(epoch, loss_contrastive.data.cpu().numpy()))
            iteration_number += 10
            counter.append(iteration_number)
            loss_history.append(loss_contrastive.data.cpu().numpy())

show_plot(counter, loss_history)

test_dataloader = DataLoader(train_data, num_workers=6, batch_size=1, shuffle=True)
dataiter = iter(test_dataloader)
x0, _, _ = next(dataiter)

for i in range(10):
    _, x1, label2 = next(dataiter)
    concatenated = torch.cat((x0, x1), 0)

    output1, output2 = net(Variable(x0).cuda(), Variable(x1).cuda())
    euclidean_distance = F.pairwise_distance(output1, output2)
    imshow(torchvision.utils.make_grid(concatenated), 'Dissimilarity: {:.2f}'.format(euclidean_distance.item()))
