import json
import numpy as np
import os
from collections import defaultdict
import torch
from torch.utils import data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
from PIL import Image
import random
import math


class CIFAR10_truncated(data.Dataset):

    def __init__(self, dataset, dataidxs=None, train=True, transform=None, target_transform=None, download=False):

        self.dataset = dataset
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):

        # cifar_dataobj = CIFAR10(self.root, self.train, self.transform, self.target_transform, self.download)
        cifar_dataobj = self.dataset

        if self.train:
            # print("train member of the class: {}".format(self.train))
            # data = cifar_dataobj.train_data

            # data = cifar_dataobj.data
            # target = np.array(cifar_dataobj.targets)

            # for i in range(len(cifar_dataobj)):
            #     if i == 0:
            #         data = np.array(cifar_dataobj[i]['x'])
            #         target = np.array(cifar_dataobj[i]['y'])
            #     else:
            #         data = np.concatenate((data, np.array(cifar_dataobj[i]['x'])),axis=0)
            #         target = np.concatenate((target, np.array(cifar_dataobj[i]['y'])),axis=0)
            #         # target = np.append(cifar_dataobj[i]['y'])
            data = np.array(cifar_dataobj['x'])
            target = np.array(cifar_dataobj['y'])

        else:
            # data = cifar_dataobj.data
            # target = np.array(cifar_dataobj.targets)
            # data = []
            # target = []
            #
            # for i in range(len(cifar_dataobj)):
            #     data.append(cifar_dataobj[i]['x'])
            #     target.append(cifar_dataobj[i]['y'])

            # for i in range(len(cifar_dataobj)):
            #
            #     data = np.vstack(cifar_dataobj[i]['x'])
            #     target = np.vstack(cifar_dataobj[i]['y'])
            data = np.array(cifar_dataobj['x'])
            target = np.array(cifar_dataobj['y'])

            # for i in range(len(cifar_dataobj)):
            #     if i == 0:
            #         data = np.array(cifar_dataobj[i]['x'])
            #         target = np.array(cifar_dataobj[i]['y'])
            #     else:
            #         data = np.concatenate((data, np.array(cifar_dataobj[i]['x'])),axis=0)
            #         target = np.concatenate((target, np.array(cifar_dataobj[i]['y'])),axis=0)

        target = target.tolist()

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]

        if self.transform is not None:
            # list to PIL Image

            img = _load_image(self.train, img)
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


def _load_image(train_or_test, img_name):
    data_dir_modify_here_1 = 'train_need_data'

    data_dir_modify_here_2 = 'test_need_data'

    if train_or_test:

        IMAGES_DIR = os.path.join('./', 'data', data_dir_modify_here_1)
        img = Image.open(os.path.join(IMAGES_DIR, img_name))
        img = img.resize((84, 84)).convert('RGB')
        return np.array(img)
    else:
        IMAGES_DIR = os.path.join('./', 'data', data_dir_modify_here_2)
        img = Image.open(os.path.join(IMAGES_DIR, img_name))
        img = img.resize((84, 84)).convert('RGB')
        return np.array(img)


def read_dir(data_dir):
    clients = []
    groups = []
    data = defaultdict(lambda: None)

    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]
    for f in files:
        file_path = os.path.join(data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        data.update(cdata['user_data'])

    clients = list(sorted(data.keys()))
    return clients, groups, data


def read_data(train_data_dir, test_data_dir):
    '''parses data in given train and test data directories

    assumes:
    - the data in the input directories are .json files with
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users

    Return:
        clients: list of client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    '''
    train_clients, train_groups, train_data = read_dir(train_data_dir)
    test_clients, test_groups, test_data = read_dir(test_data_dir)

    assert train_clients == test_clients
    assert train_groups == test_groups

    return train_clients, train_groups, train_data, test_data


def read_cifar_dir(data_dir):
    clients = []
    groups = []

    data = defaultdict(lambda: None)
    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]
    for f in files:
        f_name = f.split("_")[1]
        client = f_name.split('.')[0]
        # print('client:' + str(client))
        clients.extend(client)
        file_path = os.path.join(data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        if 'hierarchies' in cdata:
            groups.extend(cdata)
        # data.update(cdata)
        data.update({str(client): cdata})

    clients = list(sorted(data.keys()))

    return clients, groups, data


def read_cifar_data(train_data_dir, test_data_dir):
    train_clients, train_groups, train_data = read_cifar_dir(train_data_dir)
    test_clients, test_groups, test_data = read_cifar_dir(test_data_dir)

    # print('train clients:' + str(train_clients))
    # print('test clients:' + str(test_clients))

    assert train_clients == test_clients
    assert train_groups == test_groups

    return train_clients, train_groups, train_data, test_data


def cifar10_loaders(train_or_test, data={'x': [], 'y': []}):
    transform_train = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),  # 依据跟定的size，从中心进行裁剪
        # transforms.RandomHorizontalFlip(),  # 依据概率p对PIL图片进行水平翻转
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # 对数据按通道进行标准化，即先减均值，再除以标准差
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if train_or_test == 'train':

        trainset = CIFAR10_truncated(data, train=True, transform=transform_train)
        data_leaf_loader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=0)

    elif train_or_test == 'test':
        testset = CIFAR10_truncated(data, train=False, transform=transform_test)
        data_leaf_loader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=0)

    return data_leaf_loader


