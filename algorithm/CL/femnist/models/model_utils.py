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
        femnist_dataobj = self.dataset

        if self.train:
            train_data_x=[]
            train_data_y=[]

            for i in range(len(femnist_dataobj)):
                cur_x=femnist_dataobj[i]['x']
                cur_y=femnist_dataobj[i]['y']
                for j in range(len(cur_y)):
                    train_data_x.append(np.array(cur_x[j]).reshape(28,28))
                    train_data_y.append(cur_y[j])
            data=train_data_x
            target=train_data_y

        else:
            train_data_x=[]
            train_data_y=[]

            for i in range(len(femnist_dataobj)):
                cur_x=femnist_dataobj[i]['x']
                cur_y=femnist_dataobj[i]['y']
                for j in range(len(cur_y)):
                    train_data_x.append(np.array(cur_x[j]).reshape(28,28))
                    train_data_y.append(cur_y[j])
            data=train_data_x
            target=train_data_y

        #target = target.tolist()
        # data=data.reshape(-1, 28, 28)

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
            img = Image.fromarray((img*255).astype(np.uint8))
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

# class CIFAR10_truncated(data.Dataset):
#
#     def __init__(self, dataset, dataidxs=None, train=True, transform=None, target_transform=None, download=False):
#
#         self.dataset = dataset
#         self.dataidxs = dataidxs
#         self.train = train
#         self.transform = transform
#         self.target_transform = target_transform
#         self.download = download
#
#         self.data, self.target = self.__build_truncated_dataset__()
#
#     def __build_truncated_dataset__(self):
#
#         # cifar_dataobj = CIFAR10(self.root, self.train, self.transform, self.target_transform, self.download)
#         cifar_dataobj = self.dataset
#
#         if self.train:
#             #print("train member of the class: {}".format(self.train))
#             #data = cifar_dataobj.train_data
#
#             # data = cifar_dataobj.data
#             # target = np.array(cifar_dataobj.targets)
#
#
#             for i in range(len(cifar_dataobj)):
#                 if i == 0:
#                     data = np.array(cifar_dataobj[i]['x']).reshape(28,28)
#                     target = np.array(cifar_dataobj[i]['y']).reshape(28,28)
#                 else:
#                     data = np.concatenate((data, np.array(cifar_dataobj[i]['x']).reshape(28,28)),axis=0)
#                     target = np.concatenate((target, np.array(cifar_dataobj[i]['y']).reshape(28,28)),axis=0)
#                     # target = np.append(cifar_dataobj[i]['y'])
#
#         else:
#             # data = cifar_dataobj.data
#             # target = np.array(cifar_dataobj.targets)
#             # data = []
#             # target = []
#             #
#             # for i in range(len(cifar_dataobj)):
#             #     data.append(cifar_dataobj[i]['x'])
#             #     target.append(cifar_dataobj[i]['y'])
#
#             # for i in range(len(cifar_dataobj)):
#             #
#             #     data = np.vstack(cifar_dataobj[i]['x'])
#             #     target = np.vstack(cifar_dataobj[i]['y'])
#             for i in range(len(cifar_dataobj)):
#                 if i == 0:
#                     data = np.array(cifar_dataobj[i]['x']).reshape(28,28)
#                     target = np.array(cifar_dataobj[i]['y']).reshape(28,28)
#                 else:
#                     data = np.concatenate((data, np.array(cifar_dataobj[i]['x']).reshape(28,28)),axis=0)
#                     target = np.concatenate((target, np.array(cifar_dataobj[i]['y']).reshape(28,28)),axis=0)
#
#         target = target.tolist()
#
#         if self.dataidxs is not None:
#             data = data[self.dataidxs]
#             target = target[self.dataidxs]
#
#         return data, target
#
#     def truncate_channel(self, index):
#         for i in range(index.shape[0]):
#             gs_index = index[i]
#             self.data[gs_index, :, :, 1] = 0.0
#             self.data[gs_index, :, :, 2] = 0.0
#
#     def __getitem__(self, index):
#         """
#         Args:
#             index (int): Index
#
#         Returns:
#             tuple: (image, target) where target is index of the target class.
#         """
#         img, target = self.data[index], self.target[index]
#
#         if self.transform is not None:
#             # list to PIL Image
#             img = Image.fromarray((img*255).astype(np.uint8))
#             img = self.transform(img)
#
#         if self.target_transform is not None:
#             target = self.target_transform(target)
#
#         return img, target
#
#     def __len__(self):
#         return len(self.data)

class CIFAR10_single_batch_truncated(data.Dataset):

    def __init__(self, data_x, data_y, batchsize, dataidxs=None, train=True, transform=None, target_transform=None, download=False):

        self.data_x = data_x
        self.data_y = data_y
        self.batchsize = batchsize
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):

        # cifar_dataobj = self.dataset

        data_x = self.data_x
        data_y = self.data_y

        # trainset 训练时只随机截取一个batch的数据进行训练
        if self.train:

            # FedSGD

            data = data_x
            target = data_y
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

            data = data_x
            target = data_y

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


def batch_data(data, batch_size, seed):
    '''
    data is a dict := {'x': [numpy array], 'y': [numpy array]} (on one client)
    returns x, y, which are both numpy array of length: batch_size
    '''
    data_x = data['x']
    data_y = data['y']

    # randomly shuffle data
    np.random.seed(seed)
    rng_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(rng_state)
    np.random.shuffle(data_y)

    # loop through mini-batches
    for i in range(0, len(data_x), batch_size):
        batched_x = data_x[i:i + batch_size]
        batched_y = data_y[i:i + batch_size]
        yield (batched_x, batched_y)


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


# def cifar10_loaders(id, train_datax, train_datay, test_datax, test_datay):
#
#
#     train_data_x = torch.zeros_like(torch.tensor(np.array(train_datax).reshape(-1, 3, 32, 32), dtype=torch.float32))
#     train_data_y = torch.zeros_like(torch.tensor(np.array(train_datay), dtype=torch.int64))
#
#     if id == 0:
#         train_data_x = torch.tensor(np.array(train_datax).reshape(-1, 3, 32, 32), dtype=torch.float32)
#         train_data_y = torch.tensor(np.array(train_datay), dtype=torch.int64)
#
#     # elif id > 0:
#     #     train_data_x = torch.cat((train_data_x, torch.tensor(np.array(train_datax).reshape(-1, 3, 32, 32), dtype=torch.float32)), 0)
#     #     train_data_y = torch.cat((train_data_y, torch.tensor(np.array(train_datay), dtype=torch.int64)), 0)
#
#     test_data_x = torch.zeros_like(torch.tensor(np.array(test_datax), dtype=torch.float32))
#     test_data_y = torch.zeros_like(torch.tensor(np.array(test_datay), dtype=torch.int64))
#     if id == 0:
#         test_data_x = torch.tensor(np.array(test_datax), dtype=torch.float32)
#         test_data_y = torch.tensor(np.array(test_datay), dtype=torch.int64)
#     # elif id > 0:
#     #     test_data_x = torch.cat((test_data_x, torch.tensor(np.array(test_datax), dtype=torch.float32)), 0)
#     #     test_data_y = torch.cat((test_data_y, torch.tensor(np.array(test_datay), dtype=torch.int64)), 0)
#
#
#     # x = torch.reshape(torch.tensor(data['x'], dtype=torch.float), (-1, 3, 32, 32))
#     # y = torch.tensor(data['y'], dtype=torch.int64)
#
#     # print(x.size())
#     # x = x.transpose((0, 2, 3, 1))  # convert to HWC
#
#     import torch.utils.data as Data
#
#     # x = x.type(torch.FloatTensor)
#     # y = y.type(torch.FloatTensor)
#
#     train_dataset = Data.TensorDataset(train_data_x, train_data_y)
#     test_dataset = Data.TensorDataset(test_data_x, test_data_y)
#
#
#     data_train_loader = Data.DataLoader(
#         dataset=train_dataset,
#         batch_size=128,
#         shuffle=True,
#         num_workers=2,
#         pin_memory=True
#     )
#
#     data_test_loader = Data.DataLoader(
#         dataset=test_dataset,
#         batch_size=100,
#         shuffle=False,
#         num_workers=2,
#         pin_memory=True
#     )
#
#     return data_train_loader, data_test_loader


# def cifar10_loaders(train_or_test, data={'x' : [],'y' : []}):
#     #
#     # # for i in range(len(data)):
#     # #     if i == 0:
#     # #         x = torch.tensor(data['' + str(i+1) + '']['x'], dtype=torch.int64)
#     # #         y = torch.tensor(data['' + str(i+1) + '']['y'], dtype=torch.int64)
#     # #     elif i > 0:
#     # #         x = torch.cat((x, torch.tensor(data['' + str(i+1) + '']['x'], dtype=torch.int64)), 0)
#     # #         y = torch.cat((y, torch.tensor(data['' + str(i+1) + '']['y'], dtype=torch.int64)), 0)
#     #
#     # # for id in range(len(data)):
#     # #     if id == 0:
#     # #         # x = torch.reshape(torch.tensor(data[0]['x']), (-1, 3, 32, 32))
#     # #         x = torch.tensor(np.array(data[0]['x']).transpose(0,3,1,2), dtype=torch.float)
#     # #         y = torch.tensor(data[0]['y'], dtype=torch.int64)
#     # #     elif id > 0:
#     # #         # x = torch.cat((x, torch.reshape(torch.tensor(data[id]['x']), (-1, 3, 32, 32))), 0)
#     # #         new_x = torch.tensor(np.array(data[id]['x']).transpose(0,3,1,2), dtype=torch.float)
#     # #         x = torch.cat((x, new_x), 0)
#     # #         y = torch.cat((y, torch.tensor(data[id]['y'], dtype=torch.int64)), 0)
#     #
#     # for id in range(len(data)):
#     #     if id == 0:
#     #         # x = torch.reshape(torch.tensor(data[0]['x']), (-1, 3, 32, 32))
#     #         x = np.array(data[0]['x']).transpose(0, 3, 1, 2)
#     #         x = torch.from_numpy(x).float().div(255)
#     #         y = torch.tensor(data[0]['y'], dtype=torch.int64)
#     #     elif id > 0:
#     #         # x = torch.cat((x, torch.reshape(torch.tensor(data[id]['x']), (-1, 3, 32, 32))), 0)
#     #
#     #         new_x = np.array(data[id]['x']).transpose(0, 3, 1, 2)
#     #         new_x = torch.from_numpy(new_x).float().div(255)
#     #
#     #         x = torch.cat((x, new_x), 0)
#     #         y = torch.cat((y, torch.tensor(data[id]['y'], dtype=torch.int64)), 0)
#     #
#     # # for id in range(len(data)):
#     # #     if id == 0:
#     # #         # x = torch.tensor(data[0]['x'], dtype=torch.float) # [128, 32, 32, 3]
#     # #
#     # #         # x = np.vstack(data[0]['x']).reshape(-1, 3, 32, 32)
#     # #
#     # #         # x = torch.tensor(np.vstack(data[0]['x']).reshape(-1, 3, 32, 32), dtype=torch.float) # [128, 32, 32, 3]
#     # #         x = torch.tensor(np.vstack(data[0]['x']).transpose(0, 3, 1, 2), dtype=torch.float) # [128, 32, 32, 3]
#     # #
#     # #         y = torch.tensor(data[0]['y'], dtype=torch.int64)
#     # #     elif id > 0:
#     # #         # new_x = np.vstack(data[id]['x']).reshape(-1, 3, 32, 32)
#     # #
#     # #         # new_x = torch.tensor(np.vstack(data[id]['x']).reshape(-1, 3, 32, 32), dtype=torch.float)
#     # #         new_x = torch.tensor(np.vstack(data[id]['x']).transpose(0, 3, 1, 2), dtype=torch.float)
#     # #
#     # #         x = torch.cat((x, new_x), 0)
#     # #         y = torch.cat((y, torch.tensor(data[id]['y'], dtype=torch.int64)), 0)
#     #
#     #
#     # # x = torch.reshape(torch.tensor(data['x'], dtype=torch.float), (-1, 3, 32, 32))
#     # # y = torch.tensor(data['y'], dtype=torch.int64)
#     #
#     # # print(x.size())
#     # # x = x.transpose((0, 2, 3, 1))  # convert to HWC
#     #
#     # import torch.utils.data as Data
#     #
#     # if train_or_test == 'train':
#     #     batch_size = 128
#     # elif train_or_test == 'test':
#     #     batch_size = 100
#     #     # batch_size = len(y)
#     #
#     # # x = x.type(torch.FloatTensor)
#     # # y = y.type(torch.FloatTensor)
#     #
#     # x = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))(x)
#     # # x = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(x)
#     # # x = transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))(x)
#     #
#     # torch_dataset = Data.TensorDataset(x, y)
#
#     for id in range(len(data)):
#         if id == 0:
#             # x = torch.reshape(torch.tensor(data[0]['x']), (-1, 3, 32, 32))
#             x = np.array(data[0]['x']).transpose(0, 3, 1, 2)
#             x = torch.from_numpy(x).float().div(255)
#             # x = torch.from_numpy(x).float().div(255).unsqueeze(0)
#             # x = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))(x)
#
#             mean = torch.tensor([0.4914, 0.4822, 0.4465], dtype=torch.float)
#             std = torch.tensor([0.2023, 0.1994, 0.2010], dtype=torch.float)
#
#             for i in range(3):
#                 x[i, :, :] = (x[i, :, :] - mean[i]) / std[i]
#
#             y = torch.tensor(data[0]['y'], dtype=torch.int64)
#         elif id > 0:
#             # x = torch.cat((x, torch.reshape(torch.tensor(data[id]['x']), (-1, 3, 32, 32))), 0)
#
#             new_x = np.array(data[id]['x']).transpose(0, 3, 1, 2)
#             new_x = torch.from_numpy(new_x).float().div(255)
#             # new_x = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))(new_x)
#
#             mean = torch.tensor([0.4914, 0.4822, 0.4465], dtype=torch.float)
#             std = torch.tensor([0.2023, 0.1994, 0.2010], dtype=torch.float)
#
#             for i in range(3):
#                 new_x[i, :, :] = (new_x[i, :, :] - mean[i]) / std[i]
#
#             x = torch.cat((x, new_x), 0)
#             y = torch.cat((y, torch.tensor(data[id]['y'], dtype=torch.int64)), 0)
#
#
#     import torch.utils.data as Data
#
#     if train_or_test == 'train':
#         batch_size = 128
#     elif train_or_test == 'test':
#         batch_size = 100
#         # batch_size = len(y)
#
#     torch_dataset = Data.TensorDataset(x, y)
#
#     if train_or_test == 'train':
#         data_leaf_loader = Data.DataLoader(
#             dataset=torch_dataset,
#             batch_size=batch_size,
#             shuffle=True,
#             num_workers=2,
#             pin_memory=True,
#         )
#     elif train_or_test == 'test':
#         data_leaf_loader = Data.DataLoader(
#             dataset=torch_dataset,
#             batch_size=batch_size,
#             shuffle=False,
#             num_workers=2,
#             pin_memory=True,
#         )
#
#     return data_leaf_loader

def femnist_loaders(train_or_test, data={'x' : [],'y' : []}):

    transform_train = transforms.Compose([

        #transforms.RandomCrop(32, padding=4),  # 依据跟定的size，从中心进行裁剪
        #transforms.RandomHorizontalFlip(),  # 依据概率p对PIL图片进行水平翻转
        transforms.ToTensor(),
        #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # 对数据按通道进行标准化，即先减均值，再除以标准差
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    #default=128,100
    if train_or_test == 'train':
        trainset = CIFAR10_truncated(data, train=True, transform=transform_train)
        data_leaf_loader = torch.utils.data.DataLoader(trainset, batch_size=640, shuffle=True, num_workers=0)

    elif train_or_test == 'test':
        testset = CIFAR10_truncated(data, train=False, transform=transform_test)
        data_leaf_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)

    return data_leaf_loader


# def femnist_loaders(train_or_test, data={'x': [], 'y': []}, test_data={'x': [], 'y': []}):
#     y = []
#     # for each_data in data:
#     for i in range(len(data)):
#
#         for j in range(len(data[i]['x'])):
#             if (i==0)and(j == 0):
#                 x = torch.reshape(torch.tensor(data[0]['x'][0], dtype=torch.float), (-1, 1, 28, 28))
#                 y.append(data[0]['y'][0])
#             else:
#                 x = torch.cat((x, torch.reshape(torch.tensor(data[i]['x'][j], dtype=torch.float), (-1, 1, 28, 28))), 0)
#                 y.append(data[i]['y'][j])
#         # print(each_data['x'])
#         # all_data_x = dict.update(each_data['x'])
#         # all_data_y = dict.update(each_data['y'])
#
#         # x = torch.reshape(torch.tensor(each_data['x']), (-1, 1, 28, 28))
#         # # x = torch.tensor(each_data['x'])
#         # y = torch.tensor(each_data['y'], dtype=torch.int64)
#     # x = torch.reshape(torch.tensor(all_data_x), (1, 1, 28, 28))
#     # y = torch.tensor(all_data_y, dtype=torch.int64)
#
#     y = torch.tensor(y, dtype=torch.int64)
#
#     import torch.utils.data as Data
#
#     batch_size = 10
#     torch_dataset = Data.TensorDataset(x, y)
#     # # ratio = 0.005
#     # ratio = 0.1
#     #
#     # num_of_each_class_train = int(len(torch_dataset) // 10 * ratio)
#     # # num_of_each_class_test = int(len(mnist_test)//10*ratio)
#     #
#     # class_idx_train = [(y == _).nonzero().numpy().squeeze() for _ in range(62)]
#     #
#     # for i in range(len(class_idx_train)):
#     #     class_idx_train[i] = class_idx_train[i][:num_of_each_class_train]
#     #     # class_idx_test[i] = class_idx_test[i][:num_of_each_class_test]
#     #
#     # torch_dataset = Data.Subset(torch_dataset, [y for z in class_idx_train for y in z])
#
#     train_loader = Data.DataLoader(
#         dataset=torch_dataset,
#         batch_size=batch_size,
#         shuffle=True,
#         num_workers=0,
#         pin_memory=True
#     )
#
#     y = []
#     # for each_data in data:
#     for i in range(len(test_data)):
#
#         for j in range(len(test_data[i]['x'])):
#             if i==0 and j == 0:
#                 x = torch.reshape(torch.tensor(test_data[i]['x'][0], dtype=torch.float), (-1, 1, 28, 28))
#                 y.append(test_data[i]['y'][0])
#             else:
#                 x = torch.cat((x, torch.reshape(torch.tensor(test_data[i]['x'][j], dtype=torch.float), (-1, 1, 28, 28))), 0)
#                 y.append(test_data[i]['y'][j])
#         # x = torch.reshape(torch.tensor(each_data['x']), (-1, 1, 28, 28))
#         # # x = torch.tensor(each_data['x'])
#         # y = torch.tensor(each_data['y'], dtype=torch.int64)
#     # x = torch.reshape(torch.tensor(all_data_x), (1, 1, 28, 28))
#     # y = torch.tensor(all_data_y, dtype=torch.int64)
#
#     y = torch.tensor(y, dtype=torch.int64)
#
#     # batch_size = len(y)
#     # batch_size = len(y)
#     batch_size = 10
#     torch_dataset = Data.TensorDataset(x, y)
#
#     # # ratio = 0.005
#     # ratio = 0.005
#     #
#     # # only sample in training data
#     # num_of_each_class_test = int(len(torch_dataset) // 10 * ratio)
#     # # num_of_each_class_test = int(len(mnist_test)//10*ratio)
#     #
#     # class_idx_test = [(y == _).nonzero().numpy().squeeze() for _ in range(62)]
#     #
#     # for i in range(len(class_idx_test)):
#     #     class_idx_test[i] = class_idx_test[i][:num_of_each_class_test]
#     #     # class_idx_test[i] = class_idx_test[i][:num_of_each_class_test]
#     #
#     # torch_dataset = Data.Subset(torch_dataset, [y for z in class_idx_test for y in z])
#
#     test_loader = Data.DataLoader(
#         dataset=torch_dataset,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=0,
#         pin_memory=True
#     )
#
#     if train_or_test == 'train':
#         return train_loader
#
#     elif train_or_test == 'test':
#         return test_loader


# def femnist_loaders(data={'x' : [],'y' : []}, test_data={'x' : [],'y' : []} ):
#
#
#     # for each_data in data:
#     for i in range(len(data)):
#         if i == 0:
#             x = torch.reshape(torch.tensor(data[0]['x']), (-1, 1, 28, 28))
#             y = torch.tensor(data[0]['y'], dtype=torch.int64)
#         elif i > 0:
#             x = torch.cat((x, torch.reshape(torch.tensor(data[i]['x']), (-1, 1, 28, 28))), 0)
#             y = torch.cat((y, torch.tensor(data[i]['y'], dtype=torch.int64)), 0)
#
#     import torch.utils.data as Data
#
#     # batch_size = 32
#     batch_size = 640
#     torch_dataset = Data.TensorDataset(x, y)
#     # ratio = 0.005
#     # ratio = 0.1
#
#     # num_of_each_class_train = int(len(torch_dataset) // 10 * ratio)
#     # # num_of_each_class_test = int(len(mnist_test)//10*ratio)
#
#     # class_idx_train = [(y == _).nonzero().numpy().squeeze() for _ in range(62)]
#
#     # for i in range(len(class_idx_train)):
#     #     class_idx_train[i] = class_idx_train[i][:num_of_each_class_train]
#     #     # class_idx_test[i] = class_idx_test[i][:num_of_each_class_test]
#
#     # torch_dataset = Data.Subset(torch_dataset, [y for z in class_idx_train for y in z])
#
#
#     train_loader = Data.DataLoader(
#         dataset=torch_dataset,
#         batch_size=batch_size,
#         shuffle=True,
#         num_workers=0,
#         pin_memory=True
#     )
#
#     for i in range(len(test_data)):
#         # print(each_data['x'])
#         if i == 0:
#             x = torch.reshape(torch.tensor(test_data[0]['x']), (-1, 1, 28, 28))
#             y = torch.tensor(test_data[0]['y'], dtype=torch.int64)
#         elif i > 0:
#             x = torch.cat((x, torch.reshape(torch.tensor(test_data[i]['x']), (-1, 1, 28, 28))), 0)
#             y = torch.cat((y, torch.tensor(test_data[i]['y'], dtype=torch.int64)), 0)
#
#     batch_size = 100
#     torch_dataset = Data.TensorDataset(x, y)
#
#     # ratio = 0.005
#     # ratio = 0.005
#
#     # # only sample in training data
#     # num_of_each_class_test = int(len(torch_dataset) // 10 * ratio)
#     # # num_of_each_class_test = int(len(mnist_test)//10*ratio)
#
#     # class_idx_test = [(y == _).nonzero().numpy().squeeze() for _ in range(62)]
#
#     # for i in range(len(class_idx_test)):
#     #     class_idx_test[i] = class_idx_test[i][:num_of_each_class_test]
#     #     # class_idx_test[i] = class_idx_test[i][:num_of_each_class_test]
#
#     # torch_dataset = Data.Subset(torch_dataset, [y for z in class_idx_test for y in z])
#
#     test_loader = Data.DataLoader(
#         dataset=torch_dataset,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=0,
#         pin_memory=True
#     )
#
#     return train_loader, test_loader