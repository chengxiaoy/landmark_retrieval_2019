import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models
from torch.optim import lr_scheduler

import torch
from kaggle.landmark_retrieval.function import *
import numpy as np
from PIL import Image
from torchvision.transforms import transforms
import time
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import PIL.ImageOps
import linecache
import random
import copy
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def image_loader(image_name):
    im = Image.open(image_name)
    im = im.convert('RGB')
    im_size_hw = np.array(im.size[::-1])

    max_side_lengths = [1050]
    images = []
    for max_side_length in max_side_lengths:
        ratio = float(max_side_length) / np.max(im_size_hw)
        new_size = tuple(np.round(im_size_hw * ratio.astype(float)).astype(np.int32))
        # fake batch dimension required to fit network's input dimensions
        loader = transforms.Compose(
            [

                transforms.Resize(new_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        )
        image = loader(im).unsqueeze(0)
        image_cuda = image.to(device, torch.float)
        images.append(image_cuda)
    return images[0]


class SiameseNetwork(nn.Module):
    def __init__(self, pretrained=True):
        super(SiameseNetwork, self).__init__()
        self.pretrained_model = models.resnet34(pretrained=pretrained)
        self.cnn1 = nn.Sequential(*list(self.pretrained_model.children())[:-2])
        self.spp = nn.AdaptiveMaxPool2d(20)
        self.pool = rmac
        self.normal = nn.functional.normalize

    def forward_once(self, x):
        x = self.cnn1(x)
        print(x.shape)
        x = self.pool(x)
        return self.normal(x, ).squeeze(-1).squeeze(-1)
        # return self.normal(x, )

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        return output1, output2

    def __repr__(self):
        return self.__class__.__name__ + '(resnet34+rmac+l2n)'


class ContrastiveLoss(nn.Module):
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


class Config():
    root = '/Users/tezign/PycharmProjects/abracadabra/cy_cv/finetuning/'
    train_data_dirs = ["../valid_data_h_risk/", "../valid_data_l_risk/"]
    test_data_dirs = [""]
    train_txt = '/Users/tezign/PycharmProjects/CV_learn/kaggle/landmark_retrieval/' \
                'annotations_landmarks/annotation_clean_train.txt'
    test_txt = '/Users/tezign/PycharmProjects/CV_learn/kaggle/landmark_retrieval/' \
               'annotations_landmarks/annotation_clean_val.txt'
    train_batch_size = 32
    test_batch_size = 16
    train_number_epochs = 1000


def get_label(file_path):
    if file_path.__contains__('-'):
        return file_path.split("/")[-1].split(".")[0].split('-')[0].strip()
    elif file_path.__contains__("\\"):
        return file_path.split('/')[-1].split(".")[0].split('\\')[1].strip()
    elif file_path.__contains__('_'):
        return file_path.split('/')[-1].split('.')[0].split('_')[0].split()
    else:
        return 0


def get_label_dict_from_dir():
    label_dict = {}
    for train_data_dir in Config.train_data_dirs:
        for dir in ['A', 'B']:
            train_dir = train_data_dir + dir + os.path.sep
            for filename in os.listdir(train_dir):
                label = get_label(filename)
                if label_dict.__contains__(label):
                    label_dict.get(label).append(train_dir + filename)
                else:
                    label_dict[label] = [train_dir + filename]
    return label_dict


def get_label_dict_from_txt(txt_file):
    f = open(txt_file, 'r')
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


train_dict = get_label_dict_from_txt(Config.train_txt)
val_dict = get_label_dict_from_txt(Config.test_txt)


class MyDataset(Dataset):

    def __init__(self, dict, transform=None, target_transform=None, should_invert=False):
        self.transform = transform
        self.target_transform = target_transform
        self.should_invert = should_invert
        self.dict = dict
        self.length = self.get_dict_value_length()

    def get_dict_value_length(self):
        length = 0
        for key in self.dict:
            length = length + len(self.dict[key])
        return length

    def __getitem__(self, index):
        while True:
            choice_label = random.choice(list(self.dict.keys()))
            files = self.dict[choice_label]
            if len(files) > 2:
                break

        should_get_same_class = random.randint(0, 1)
        if should_get_same_class:
            file1, file2 = random.sample(files, k=2)
            img0 = image_loader(file1)
            img1 = image_loader(file2)
            return img0, img1, torch.from_numpy(np.array([False], dtype=np.float32))
        else:
            choice_label_2 = random.choice(list(self.dict.keys()))
            while choice_label_2 == choice_label:
                choice_label_2 = random.choice(list(self.dict.keys()))
            files_2 = self.dict[choice_label_2]
            file1 = random.choice(files)
            file2 = random.choice(files_2)
            img0 = image_loader(file1)
            img1 = image_loader(file2)
            return img0, img1, torch.from_numpy(np.array([True], dtype=np.float32))

    def __len__(self):
        return self.length


# Training


def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=100):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    min_loss = float('inf')

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over version5_gray_data_2W_top3-0.7.
            for i, data in enumerate(dataloaders[phase]):
                img0, img1, label = data
                img0, img1, label = img0.to(device), img1.to(device), label.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    output1, output2 = model(img0, img1)

                    loss_contrastive = criterion(output1, output2, label)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss_contrastive.backward()
                        optimizer.step()

                # statistics
                running_loss += loss_contrastive.item()

            epoch_loss = running_loss / dataset_sizes[phase]

            print('{} Loss: {:.4f} '.format(phase, epoch_loss))

            # deep copy the model
            if phase == 'val' and epoch_loss < min_loss:
                min_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('min loss : {:4f}'.format(min_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)

    # save model
    torch.save(model.state_dict(), str(model) + ".pth")

    return model


class siames_model:
    def __init__(self):
        pass

    def fine_tune_pretrained_model(self):
        train_data = MyDataset(train_dict, should_invert=False)
        train_dataloader = DataLoader(dataset=train_data, shuffle=True, num_workers=4,
                                      batch_size=Config.train_batch_size)
        test_data = MyDataset(val_dict
                              , should_invert=False)
        test_dataloader = DataLoader(dataset=test_data, shuffle=True, num_workers=4, batch_size=Config.test_batch_size)
        net = SiameseNetwork().to(device)
        criterion = ContrastiveLoss()
        optimizer = optim.Adam(net.parameters(), lr=0.0005)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
        dataloaders = {"train": train_dataloader, "val": test_dataloader}
        dataset_sizes = {"train": len(train_data), "val": len(test_data)}
        train_model(net, criterion, optimizer, exp_lr_scheduler, dataloaders, dataset_sizes, 1000)

    def extract_feature(self, image_path, finetuning=False):
        img = image_loader(image_path)
        if not finetuning:
            net = SiameseNetwork().to(device).eval()
        else:
            net = SiameseNetwork(False).to(device)
            net.load_state_dict(torch.load(str(net) + ".pth"))
            net.eval()
        feature = net.forward_once(img).data.cpu().numpy()
        print(feature)
        return feature


if __name__ == '__main__':
    model = siames_model()
    since = time.time()
    model.fine_tune_pretrained_model()
    model.extract_feature("404.jpg")
