import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models
from torch.optim import lr_scheduler

from function import *
import numpy as np
from PIL import Image
from torchvision.transforms import transforms
import time
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import random
import copy
import os
from sklearn.preprocessing import normalize as sknormalize

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def image_loader(image_name):
    im = Image.open(image_name)
    im = im.convert('RGB')
    im_size_hw = np.array(im.size[::-1])

    max_side_lengths = [512]
    images = []
    for max_side_length in max_side_lengths:
        ratio = float(max_side_length) / np.max(im_size_hw)
        new_size = tuple(np.round(im_size_hw * ratio.astype(float)).astype(np.int32))
        # fake batch dimension required to fit network's input dimensions
        loader = transforms.Compose(
            [

                transforms.Resize(new_size),
                # transforms.RandomHorizontalFlip(),
                # transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        )
        image = loader(im).unsqueeze(0)
        # image_cuda = image.to(device, torch.float)
        images.append(image)
    return images[0]


def image_loader_eval(image_name):
    im = Image.open(image_name)
    im = im.convert('RGB')
    im_size_hw = np.array(im.size[::-1])

    max_side_lengths = [550, 800, 1050]
    images = []
    for max_side_length in max_side_lengths:
        ratio = float(max_side_length) / np.max(im_size_hw)
        new_size = tuple(np.round(im_size_hw * ratio.astype(float)).astype(np.int32))
        # fake batch dimension required to fit network's input dimensions
        loader = transforms.Compose(
            [
                # transforms.Grayscale(num_output_channels=3),
                transforms.Resize(new_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        )
        image = loader(im).unsqueeze(0)
        images.append(image)
    return images


class SiameseNetwork(nn.Module):
    def __init__(self, pretrained=True):
        super(SiameseNetwork, self).__init__()
        self.pretrained_model = models.resnet50(pretrained=pretrained)
        self.cnn1 = nn.Sequential(*list(self.pretrained_model.children())[:-2])
        self.pool = gem
        self.normal = nn.functional.normalize

    def forward_once(self, x):
        x = self.cnn1(x)
        x = self.pool(x)
        return self.normal(x, ).squeeze(-1).squeeze(-1)

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        return output1, output2

    def __repr__(self):
        return self.__class__.__name__ + '(resnet50_gem_eval)'


class ContrastiveLossNew(nn.Module):
    r"""CONTRASTIVELOSS layer that computes contrastive loss for a batch of images:
        Q query tuples, each packed in the form of (q,p,n1,..nN)

    Args:
        x: tuples arranges in columns as [q,p,n1,nN, ... ]
        label: -1 for query, 1 for corresponding positive, 0 for corresponding negative
        margin: contrastive loss margin. Default: 0.7

    >>> contrastive_loss = ContrastiveLossNew(margin=0.7)
    >>> input = torch.randn(128, 35, requires_grad=True)
    >>> label = torch.Tensor([-1, 1, 0, 0, 0, 0, 0] * 5)
    >>> output = contrastive_loss(input, label)
    >>> output.backward()
    """

    def __init__(self, margin=0.7, eps=1e-6):
        super(ContrastiveLossNew, self).__init__()
        self.margin = margin
        self.eps = eps

    def forward(self, x, label):
        return contrastive_loss(x, label, margin=self.margin, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'margin=' + '{:.4f}'.format(self.margin) + ')'


class Config():
    train_txt = 'annotation_clean_train.txt'
    test_txt = 'annotation_clean_val.txt'
    train_batch_size = 5
    test_batch_size = 5
    train_number_epochs = 100


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


class TupleDataset(Dataset):
    def __init__(self, dict, nq=5):
        """

        :param dict:
        :param nq: negative nums
        """
        self.dict = dict
        self.key_list = list(dict.keys())
        self.nq = nq

    def __getitem__(self, index):
        """
        return one query + 1 positive + n negative
        :param index:
        :return:
        """
        key = self.key_list[index]
        query, positive = random.sample(self.dict[key], k=2)
        items = []
        items.append(query)
        items.append(positive)
        for i in range(self.nq):
            n_key = random.choice(self.key_list)
            while n_key == key:
                n_key = random.choice(self.key_list)
            items.append(random.choice(self.dict[n_key]))

        target = torch.Tensor([-1, 1] + [0] * self.nq)

        res = []
        for item in items:
            res.append(image_loader(item))
        return res, target

    def __len__(self):
        return len(self.key_list)


# Training


def train_model(model, criterion, optimizer, scheduler, dataloaders, num_epochs=100):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    min_loss = float('inf')

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            # model.eval()
            if phase == 'train':
                # scheduler.step()
                model.train()  # Set model to training mode
                model.apply(set_batchnorm_eval)
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            # Iterate over version5_gray_data_2W_top3-0.7.
            for i, (input, target) in enumerate(dataloaders[phase]):
                nq = len(input)  # number of training tuples
                ni = len(input[0])  # number of images per tuple
                optimizer.zero_grad()
                for q in range(nq):
                    # output = torch.zeros(model.meta['outputdim'], ni).cuda()
                    output = torch.zeros(2048, ni).cuda()
                    for imi in range(ni):
                        # compute output vector for image imi
                        # output[:, imi] = model(input[q][imi].cuda()).squeeze()
                        output[:, imi] = model.forward_once(input[q][imi].cuda())

                    # reducing memory consumption:
                    # compute loss for this query tuple only
                    # then, do backward pass for one tuple only
                    # each backward pass gradients will be accumulated
                    # the optimization step is performed for the full batch later
                    loss = criterion(output, target[q].cuda())
                    if phase == 'train':
                        loss.backward()
                    # print("phase {} , {} batch".format(phase, i))
                    batch_loss = loss.data.cpu().item()
                    # print("loss is {}".format(batch_loss))
                    running_loss += batch_loss
                optimizer.step()
            # 5 is batch_size
            # print("size {}",format(len(dataloaders[phase])))

            epoch_loss = running_loss / (len(dataloaders[phase]) * 5)

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


def collate_triples(batch):
    if len(batch) == 1:
        return [batch[0][0]], [batch[0][1]]
    return [batch[i][0] for i in range(len(batch))], [batch[i][1] for i in range(len(batch))]


class siames_model:
    def __init__(self, file_path, finetuning=True):
        if not finetuning:
            self.net = SiameseNetwork().to(device).eval()
        else:
            print("load finetuing model {}".format(file_path))
            net = SiameseNetwork(False).to(device)
            net.load_state_dict(torch.load(file_path))
            self.net = net.eval()

    def normalize(self, x, copy=False):
        """
        A helper function that wraps the function of the same name in sklearn.
        This helper handles the case of a single column vector.
        """
        if type(x) == np.ndarray and len(x.shape) == 1:
            return np.squeeze(sknormalize(x.reshape(1, -1), copy=copy))
        else:
            return sknormalize(x, copy=copy)

    def fine_tune_pretrained_model(self):
        train_dict = get_label_dict_from_txt(Config.train_txt)
        val_dict = get_label_dict_from_txt(Config.test_txt)

        train_dict = self.filter_dict(train_dict, 10)

        val_dict = self.filter_dict(val_dict, 10)

        train_data = TupleDataset(train_dict)
        train_dataloader = DataLoader(dataset=train_data, shuffle=True, num_workers=4,
                                      batch_size=Config.train_batch_size, collate_fn=collate_triples)
        test_data = TupleDataset(val_dict)
        test_dataloader = DataLoader(dataset=test_data, shuffle=True, num_workers=4, batch_size=Config.test_batch_size,
                                     collate_fn=collate_triples)
        net = SiameseNetwork().to(device)
        criterion = ContrastiveLossNew()
        optimizer = optim.Adam(net.parameters(), lr=0.0000005, weight_decay=0.0003)
        exp_decay = math.exp(-0.01)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=exp_decay)
        # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        dataloaders = {"train": train_dataloader, "val": test_dataloader}
        train_model(net, criterion, optimizer, scheduler, dataloaders, Config.train_number_epochs)

    def extract_feature(self, image_path):

        img = image_loader(image_path)
        feature = self.net.forward_once(img.to(device)).data.cpu().numpy()
        return feature[0]

    def split_dict(self, dict):
        keys = []
        val_dict = {}

        i = 0
        for key in dict:
            if i < 100:
                keys.append(key)
                val_dict[key] = dict[key]
                i = i + 1
        for key in keys:
            dict.pop(key)
        return dict, val_dict

    def filter_dict(self, dict, k):
        """
        将样本少于k的标签删除
        :param dict:
        :param k:
        :return:
        """
        rm_keys = []
        for key in dict:
            if len(dict[key]) < k:
                rm_keys.append(key)
        for rm_key in rm_keys:
            dict.pop(rm_key)

        print("key size {}".format(len(dict)))
        return dict


def set_batchnorm_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        # freeze running mean and std:
        # we do training one image at a time
        # so the statistics would not be per batch
        # hence we choose freezing (ie using imagenet statistics)
        m.eval()
        # # freeze parameters:
        # # in fact no need to freeze scale and bias
        # # they can be learned
        # # that is why next two lines are commented
        # for p in m.parameters():
        # p.requires_grad = False


def valid_ft_model():
    image_tests = [[], [], []]
    ft_model = siames_model('SiameseNetwork(resnet50_gem_eval).pth', finetuning=True)
    model = siames_model("", finetuning=False)
    dis1_list = []
    dis2_list = []
    for image_test in image_tests:
        query1 = ft_model.extract_feature(image_test[0])
        pos1 = ft_model.extract_feature(image_test[1])
        nega1 = ft_model.extract_feature(image_test[2])
        dis1 = np.dot(query1, pos1) - np.dot(query1, nega1)
        dis1_list.append(dis1)
        query2 = model.extract_feature(image_test[0])
        pos2 = model.extract_feature(image_test[1])
        nega2 = model.extract_feature(image_test[2])
        dis2 = np.dot(query2, pos2) - np.dot(query2, nega2)
        dis2_list.append(dis2)
    print(dis1_list)
    print(dis2_list)


if __name__ == '__main__':
    model = siames_model('SiameseNetwork(resnet50_gem_eval).pth', finetuning=True)
    print(str(model.net))
    feature = model.extract_feature("test.jpg")
    print(feature)
    print(np.dot(feature, feature.T))
    # since = time.time()
    # model.fine_tune_pretrained_model()
    # print("fine-tuning used {} s".format(str(time.time() - since)))
