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

        # trainset 训练时只随机截取一个batch的数据进行训练
        if self.train:

            # FedSGD

            large_num = len(cifar_dataobj['x']) - self.batchsize - 1
            random_num = random.randint(1, large_num)  # 一个client图片有500张
            data = np.array(cifar_dataobj['x'][random_num: random_num + self.batchsize])
            target = np.array(cifar_dataobj['y'][random_num: random_num + self.batchsize])
            # print(len(data))
            # print(data)

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
            # random_num = random.randint(1, 100)  # 一个client图片有500张
            # batch_num = (len(cifar_dataobj['x']) / self.batchsize) / 2  # batch_size=32
            # batch_num = math.ceil(batch_num)
            # data = np.array(cifar_dataobj['x'][0: 0 + batch_num * self.batchsize])
            # target = np.array(cifar_dataobj['y'][0: 0 + batch_num * self.batchsize])

            # random_num = random.randint(1, 100)  # 一个client图片有300-600张

            # batchsize_changenum = (len(cifar_dataobj['x']) / self.batchsize) / 3 # (len(cifar_dataobj['x']) / self.batchsize)值为10-20
            # batchsize = math.ceil(self.batchsize * batchsize_changenum)

            # batchsize = math.ceil(len(cifar_dataobj['x']) / 3)# (len(cifar_dataobj['x']) / self.batchsize)值为10-20
            batchsize = math.ceil(len(cifar_dataobj['x']) / 15.625)  # 总batchsize=640 32*20
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


# def batch_data(data, batch_size, seed):
#     '''
#     data is a dict := {'x': [numpy array], 'y': [numpy array]} (on one client)
#     returns x, y, which are both numpy array of length: batch_size
#     '''
#     data_x = data['x']
#     data_y = data['y']
#
#     # randomly shuffle data
#     np.random.seed(seed)
#     rng_state = np.random.get_state()
#     np.random.shuffle(data_x)
#     np.random.set_state(rng_state)
#     np.random.shuffle(data_y)
#
#     # loop through mini-batches
#     for i in range(0, len(data_x), batch_size):
#         batched_x = data_x[i:i + batch_size]
#         batched_y = data_y[i:i + batch_size]
#         yield (batched_x, batched_y)

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


# def cifar10_loaders(train_or_test, data={'x' : [],'y' : []}):
#
#     # for i in range(len(data)):
#     #     if i == 0:
#     #         x = torch.tensor(data['' + str(i+1) + '']['x'], dtype=torch.int64)
#     #         y = torch.tensor(data['' + str(i+1) + '']['y'], dtype=torch.int64)
#     #     elif i > 0:
#     #         x = torch.cat((x, torch.tensor(data['' + str(i+1) + '']['x'], dtype=torch.int64)), 0)
#     #         y = torch.cat((y, torch.tensor(data['' + str(i+1) + '']['y'], dtype=torch.int64)), 0)
#
#     # x = torch.tensor(data['x'], dtype=torch.float)
#     # x = torch.reshape(torch.tensor(data['x'], dtype=torch.float), (-1, 3, 32, 32))
#     x = torch.tensor(np.array(data['x']).transpose(0,3,1,2), dtype=torch.float)
#     y = torch.tensor(data['y'], dtype=torch.int64)
#
#     # print(x.size())
#
#     import torch.utils.data as Data
#
#     if train_or_test == 'train':
#         # batch_size = 128
#         batch_size = 10
#         shuffle = True
#
#     elif train_or_test == 'test':
#         # batch_size = 100
#         batch_size = 10
#         shuffle = False
#
#     torch_dataset = Data.TensorDataset(x, y)
#
#     data_leaf_loader = Data.DataLoader(
#         dataset=torch_dataset,
#         batch_size=batch_size,
#         shuffle=shuffle,
#         num_workers=2,
#         pin_memory=True
#     )
#
#     return data_leaf_loader


def cifar10_loaders(train_or_test, data={'x': [], 'y': []}, test_data={'x': [], 'y': []}):
    y = []
    # for each_data in data:
    for i in range(len(data)):
        # print(each_data['x'])
        # all_data_x = dict.update(each_data['x'])
        # all_data_y = dict.update(each_data['y'])
        if i == 0:
            x = torch.reshape(torch.tensor(data[i]['x'], dtype=torch.float), (-1, 1, 28, 28))
            y.append(data[i]['y'])
        elif i > 0:
            x = torch.cat((x, torch.reshape(torch.tensor(data[i]['x'], dtype=torch.float), (-1, 1, 28, 28))), 0)
            y.append(data[i]['y'])
        # x = torch.reshape(torch.tensor(each_data['x']), (-1, 1, 28, 28))
        # # x = torch.tensor(each_data['x'])
        # y = torch.tensor(each_data['y'], dtype=torch.int64)
    # x = torch.reshape(torch.tensor(all_data_x), (1, 1, 28, 28))
    # y = torch.tensor(all_data_y, dtype=torch.int64)

    y = torch.tensor(y, dtype=torch.int64)

    import torch.utils.data as Data

    batch_size = 10
    torch_dataset = Data.TensorDataset(x, y)
    # # ratio = 0.005
    # ratio = 0.1
    #
    # num_of_each_class_train = int(len(torch_dataset) // 10 * ratio)
    # # num_of_each_class_test = int(len(mnist_test)//10*ratio)
    #
    # class_idx_train = [(y == _).nonzero().numpy().squeeze() for _ in range(62)]
    #
    # for i in range(len(class_idx_train)):
    #     class_idx_train[i] = class_idx_train[i][:num_of_each_class_train]
    #     # class_idx_test[i] = class_idx_test[i][:num_of_each_class_test]
    #
    # torch_dataset = Data.Subset(torch_dataset, [y for z in class_idx_train for y in z])

    train_loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    y = []
    # for each_data in data:
    for i in range(len(test_data)):
        # print(each_data['x'])
        # all_data_x = dict.update(each_data['x'])
        # all_data_y = dict.update(each_data['y'])
        if i == 0:
            x = torch.reshape(torch.tensor(test_data[i]['x'], dtype=torch.float), (-1, 1, 28, 28))
            y.append(test_data[i]['y'])

        elif i > 0:
            x = torch.cat((x, torch.reshape(torch.tensor(test_data[i]['x'], dtype=torch.float), (-1, 1, 28, 28))), 0)
            y.append(test_data[i]['y'])
        # x = torch.reshape(torch.tensor(each_data['x']), (-1, 1, 28, 28))
        # # x = torch.tensor(each_data['x'])
        # y = torch.tensor(each_data['y'], dtype=torch.int64)
    # x = torch.reshape(torch.tensor(all_data_x), (1, 1, 28, 28))
    # y = torch.tensor(all_data_y, dtype=torch.int64)

    y = torch.tensor(y, dtype=torch.int64)

    # batch_size = len(y)
    # batch_size = len(y)
    batch_size = 10
    torch_dataset = Data.TensorDataset(x, y)

    # # ratio = 0.005
    # ratio = 0.005
    #
    # # only sample in training data
    # num_of_each_class_test = int(len(torch_dataset) // 10 * ratio)
    # # num_of_each_class_test = int(len(mnist_test)//10*ratio)
    #
    # class_idx_test = [(y == _).nonzero().numpy().squeeze() for _ in range(62)]
    #
    # for i in range(len(class_idx_test)):
    #     class_idx_test[i] = class_idx_test[i][:num_of_each_class_test]
    #     # class_idx_test[i] = class_idx_test[i][:num_of_each_class_test]
    #
    # torch_dataset = Data.Subset(torch_dataset, [y for z in class_idx_test for y in z])

    test_loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    if train_or_test == 'train':
        return train_loader

    elif train_or_test == 'test':
        return test_loader


def cifar10_singlebatch_loaders(train_or_test, data={'x': [], 'y': []}):
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


def cifar10_somebatch_loaders(train_or_test, data={'x': [], 'y': []}):
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
        data_leaf_loader = torch.utils.data.DataLoader(trainset, batch_size=trainset.batchsize, shuffle=False,
                                                       num_workers=0)

    elif train_or_test == 'test':
        testset = CIFAR10_some_batch_truncated(data, batch_size, train=False, transform=transform_test)
        data_leaf_loader = torch.utils.data.DataLoader(testset, batch_size=testset.batchsize, shuffle=False,
                                                       num_workers=0)

    return data_leaf_loader