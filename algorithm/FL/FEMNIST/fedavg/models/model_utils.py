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

            data = np.array(cifar_dataobj['x'])
            target = np.array(cifar_dataobj['y'])

        else:

            data = np.array(cifar_dataobj['x'])
            target = np.array(cifar_dataobj['y'])


        target = target.tolist()

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def truncate_channel(self, index):
        for i in range(index.shape[0]):
            gs_index = index[i]
            self.data[gs_index, :, :, 1] = 0.0
            self.data[gs_index, :, :, 2] = 0.0

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
            img = Image.fromarray(np.uint8(img))
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

class CIFAR10_single_batch_truncated(data.Dataset):

    def __init__(self, dataset, batchsize, dataidxs=None, train=True, transform=None, target_transform=None, download=False):

        self.dataset = dataset
        self.batchsize = batchsize
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):

        cifar_dataobj = self.dataset

        # trainset 训练时只随机截取一个batch的数据进行训练
        if self.train:

            # FedSGD

            large_num = len(cifar_dataobj['x']) - self.batchsize - 1
            random_num = random.randint(1, large_num)# 一个client图片有500张
            data = np.array(cifar_dataobj['x'][random_num: random_num + self.batchsize])
            target = np.array(cifar_dataobj['y'][random_num: random_num + self.batchsize])


            # randomly shuffle data
            seed = 0
            np.random.seed(seed)
            rng_state = np.random.get_state()
            np.random.shuffle(data)
            np.random.set_state(rng_state)
            np.random.shuffle(target)

        else:

            data = np.array(cifar_dataobj['x'])
            target = np.array(cifar_dataobj['y'])

        target = target.tolist()

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def truncate_channel(self, index):
        for i in range(index.shape[0]):
            gs_index = index[i]
            self.data[gs_index, :, :, 1] = 0.0
            self.data[gs_index, :, :, 2] = 0.0

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
            img = Image.fromarray(np.uint8(img))
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


class CIFAR10_some_batch_truncated(data.Dataset):

    def __init__(self, dataset, batchsize, dataidxs=None, train=True, transform=None, target_transform=None,
                 download=False):

        self.dataset = dataset
        self.batchsize = batchsize
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):

        cifar_dataobj = self.dataset

        # trainset 训练时只随机截取batch_num个batch的数据进行训练
        if self.train:

            # batchSGD

            # batchsize = math.ceil(len(cifar_dataobj['x']) / 3)# (len(cifar_dataobj['x']) / self.batchsize)值为10-20
            batchsize = math.ceil(len(cifar_dataobj['x']) / 15.625)# 总batchsize=640 32*20
            # batchsize = math.ceil(len(cifar_dataobj['x']) / 3.90625)# 总batchsize=2560 128*20

            large_num = len(cifar_dataobj['x']) - batchsize - 1
            random_num = random.randint(1, large_num)  # 一个client图片有500张

            data = np.array(cifar_dataobj['x'][random_num: random_num + batchsize])
            target = np.array(cifar_dataobj['y'][random_num: random_num + batchsize])

            # randomly shuffle data
            seed = 0
            np.random.seed(seed)
            rng_state = np.random.get_state()
            np.random.shuffle(data)
            np.random.set_state(rng_state)
            np.random.shuffle(target)

            # print(len(data))

        else:

            data = np.array(cifar_dataobj['x'])
            target = np.array(cifar_dataobj['y'])

        target = target.tolist()

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def truncate_channel(self, index):
        for i in range(index.shape[0]):
            gs_index = index[i]
            self.data[gs_index, :, :, 1] = 0.0
            self.data[gs_index, :, :, 2] = 0.0

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
            img = Image.fromarray(np.uint8(img))
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

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


def cifar10_loaders(train_or_test, data={'x' : [],'y' : []}):

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # 依据跟定的size，从中心进行裁剪
        transforms.RandomHorizontalFlip(),  # 依据概率p对PIL图片进行水平翻转
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

def cifar10_singlebatch_loaders(train_or_test, data={'x' : [],'y' : []}):

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # 依据跟定的size，从中心进行裁剪
        transforms.RandomHorizontalFlip(),  # 依据概率p对PIL图片进行水平翻转
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # 对数据按通道进行标准化，即先减均值，再除以标准差
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    batch_size = 32
    # batch_size = 128

    if train_or_test == 'train':
        trainset = CIFAR10_single_batch_truncated(data, batch_size, train=True, transform=transform_train)
        data_leaf_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=0)

    elif train_or_test == 'test':
        testset = CIFAR10_single_batch_truncated(data, batch_size, train=False, transform=transform_test)
        data_leaf_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

    return data_leaf_loader

def cifar10_somebatch_loaders(train_or_test, data={'x' : [],'y' : []}):

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # 依据跟定的size，从中心进行裁剪
        transforms.RandomHorizontalFlip(),  # 依据概率p对PIL图片进行水平翻转
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # 对数据按通道进行标准化，即先减均值，再除以标准差
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    batch_size = 32

    if train_or_test == 'train':
        trainset = CIFAR10_some_batch_truncated(data, batch_size, train=True, transform=transform_train)
        data_leaf_loader = torch.utils.data.DataLoader(trainset, batch_size=trainset.batchsize, shuffle=False, num_workers=0)

    elif train_or_test == 'test':
        testset = CIFAR10_some_batch_truncated(data, batch_size, train=False, transform=transform_test)
        data_leaf_loader = torch.utils.data.DataLoader(testset, batch_size=testset.batchsize, shuffle=False, num_workers=0)

    return data_leaf_loader