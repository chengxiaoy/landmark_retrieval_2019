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
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
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
        # return self.normal(x, )

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        return output1, output2

    def __repr__(self):
        return self.__class__.__name__ + '(resnet50+gem+l2n+lr)'


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=0.75):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive


class ContrastiveLossNew(nn.Module):
    r"""CONTRASTIVELOSS layer that computes contrastive loss for a batch of images:
        Q query tuples, each packed in the form of (q,p,n1,..nN)

    Args:
        x: tuples arranges in columns as [q,p,n1,nN, ... ]
        label: -1 for query, 1 for corresponding positive, 0 for corresponding negative
        margin: contrastive loss margin. Default: 0.7

    >>> contrastive_loss = ContrastiveLoss(margin=0.7)
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
            if len(files) > 10:
                break

        should_get_same_class = random.randint(0, 1)
        if should_get_same_class:
            file1, file2 = random.sample(files, k=2)
            img0 = image_loader(file1)
            img1 = image_loader(file2)
            return img0, img1, torch.from_numpy(np.array([0], dtype=np.float32))
        else:
            choice_label_2 = random.choice(list(self.dict.keys()))
            while choice_label_2 == choice_label:
                choice_label_2 = random.choice(list(self.dict.keys()))
            files_2 = self.dict[choice_label_2]
            file1 = random.choice(files)
            file2 = random.choice(files_2)
            img0 = image_loader(file1)
            img1 = image_loader(file2)
            return img0, img1, torch.from_numpy(np.array([1], dtype=np.float32))

    def __len__(self):
        # return 2000
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
            max_batchs = {'train': 400, 'val': 100}
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            batchs = 0
            # Iterate over version5_gray_data_2W_top3-0.7.
            for i, data in enumerate(dataloaders[phase]):
                batchs = batchs + 1
                img0, img1, label = data
                nq = len(img0)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    for q in range(nq):
                        output1, output2 = model(img0[q].to(device), img1[q].to(device))
                        loss_contrastive = criterion(output1, output2, label[q].to(device))

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss_contrastive.backward()
                        # statistics
                        running_loss += loss_contrastive.data.cpu().item()

                    optimizer.step()
                if batchs == max_batchs[phase]:
                    break
            epoch_loss = running_loss / (batchs * 5)

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
        return [batch[0][0]], [batch[0][1]], [batch[0][2]]
    return [batch[i][0] for i in range(len(batch))], [batch[i][1] for i in range(len(batch))], [batch[i][2] for i in
                                                                                                range(len(batch))]


class siames_model:
    def __init__(self, file_path, finetuning=True):
        if not finetuning:
            self.net = SiameseNetwork().to(device).eval()
        else:
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
        train_data = MyDataset(train_dict, should_invert=False)
        train_dataloader = DataLoader(dataset=train_data, shuffle=True, num_workers=4,
                                      batch_size=Config.train_batch_size, collate_fn=collate_triples)
        test_data = MyDataset(val_dict, should_invert=False)
        test_dataloader = DataLoader(dataset=test_data, shuffle=True, num_workers=4, batch_size=Config.test_batch_size,
                                     collate_fn=collate_triples)
        net = SiameseNetwork().to(device)
        criterion = ContrastiveLoss()
        optimizer = optim.Adam(net.parameters(), lr=5e-7, weight_decay=0.0003)

        exp_decay = math.exp(-0.01)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=exp_decay)
        # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        dataloaders = {"train": train_dataloader, "val": test_dataloader}
        dataset_sizes = {"train": len(train_data), "val": len(test_data)}
        train_model(net, criterion, optimizer, scheduler, dataloaders, dataset_sizes, Config.train_number_epochs)

    def extract_feature(self, image_path):

        img = image_loader(image_path)
        feature = self.net.forward_once(img.to(device)).data.cpu().numpy()
        return feature[0]


if __name__ == '__main__':
    val_dict = get_label_dict_from_txt(Config.test_txt)
    train_dict = get_label_dict_from_txt(Config.train_txt)
    model = siames_model('resnet50.pth', finetuning=False)
    print(str(model.net))
    since = time.time()
    model.fine_tune_pretrained_model()
    print("fine-tuning used {} s".format(str(time.time() - since)))
