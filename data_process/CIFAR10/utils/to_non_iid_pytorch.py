import argparse
import json
import numpy as np
import torchvision.transforms as transforms
from datasets import MNIST_truncated, CIFAR10_truncated

import torch.nn.functional as F
from torch.autograd import Variable

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-dataset',
                        help='name of dataset;',
                        type=str,
                        choices=['cifar100', 'cifar10', 'mnist'],
                        default='cifar10')
    parser.add_argument('-total_clients', help="total number of client after federated partition", default=100) # default=50

    parser.add_argument('-datadir', help="location of origin cifar10 dataset",  default='../data/raw_data')

    return parser

def load_mnist_data(datadir):

	transform = transforms.Compose([transforms.ToTensor()])

	mnist_train_ds = MNIST_truncated(datadir, train=True, download=True, transform=transform)
	mnist_test_ds = MNIST_truncated(datadir, train=False, download=True, transform=transform)

	X_train, y_train = mnist_train_ds.data, mnist_train_ds.target
	X_test, y_test = mnist_test_ds.data, mnist_test_ds.target

	X_train = X_train.data.numpy()
	y_train = y_train.data.numpy()
	X_test = X_test.data.numpy()
	y_test = y_test.data.numpy()

	return (X_train, y_train, X_test, y_test)

def load_cifar10_data(datadir):

	# transform = transforms.Compose([transforms.ToTensor()])
    #
	# cifar10_train_ds = CIFAR10_truncated(datadir, train=True, download=True, transform=transform)
	# cifar10_test_ds = CIFAR10_truncated(datadir, train=False, download=True, transform=transform)


    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: F.pad(
            Variable(x.unsqueeze(0), requires_grad=False),
            (4, 4, 4, 4), mode='reflect').data.squeeze()),
        transforms.ToPILImage(),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    # data prep for test set
    transform_test = transforms.Compose([transforms.ToTensor(), normalize])

    cifar10_train_ds = CIFAR10_truncated(datadir, train=True, transform=transform_train, download=True)
    cifar10_test_ds = CIFAR10_truncated(datadir, train=False, transform=transform_test, download=True)

    X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.target
    X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.target

    return (X_train, y_train, X_test, y_test)

def load_cifar100_data(datadir):

    transform = transforms.Compose([transforms.ToTensor()])

    cifar100_train_ds = CIFAR100_truncated(datadir, train=True, download=True, transform=transform)
    cifar100_test_ds = CIFAR100_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = cifar100_train_ds.data, cifar100_train_ds.target
    X_test, y_test = cifar100_test_ds.data, cifar100_test_ds.target

    return (X_train, y_train, X_test, y_test)

def record_net_data_stats(y_train, net_dataidx_map):

	net_cls_counts = {}

	for net_i, dataidx in net_dataidx_map.items():
		unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
		tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
		net_cls_counts[net_i] = tmp

	return net_cls_counts

def partition_data(dataset, datadir, n_nets, alpha=0.5):

    if dataset == 'mnist':
        X_train, y_train, X_test, y_test = load_mnist_data(datadir)
    elif dataset == 'cifar10':
        X_train, y_train, X_test, y_test = load_cifar10_data(datadir)

    n_train = X_train.shape[0]

    min_size = 0
    K = 10
    N = y_train.shape[0]
    N_test = y_test.shape[0]
    net_dataidx_map = {}
    net_dataidx_map_test = {}

    while min_size < 10:
        idx_batch = [[] for _ in range(n_nets)]
        idx_batch_test = [[] for _ in range(n_nets)]
        for k in range(K):
            idx_k = np.where(y_train == k)[0]
            idx_k_test = np.where(y_test == k)[0]
            np.random.shuffle(idx_k)
            np.random.shuffle(idx_k_test)
            proportions = np.random.dirichlet(np.repeat(alpha, n_nets))
            ## Balance
            proportions = np.array([p * (len(idx_j) < N / n_nets) for p, idx_j in zip(proportions, idx_batch)])
            proportions_test = np.array([p * (len(idx_j) < N_test / n_nets) for p, idx_j in zip(proportions, idx_batch_test)])
            proportions = proportions / proportions.sum()
            proportions_test = proportions_test/proportions_test.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            proportions_test = (np.cumsum(proportions_test) * len(idx_k_test)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            idx_batch_test = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch_test, np.split(idx_k_test, proportions_test))]
            min_size = min([len(idx_j) for idx_j in idx_batch])
            min_size_test = min([len(idx_j) for idx_j in idx_batch_test])


    # if dataset == 'cifar100':
    #     X_train, y_train, X_test, y_test = load_cifar100_data(datadir)
    #
    # n_train = X_train.shape[0]
    #
    # min_size = 0
    # K = 100
    # N = y_train.shape[0]
    # N_test = y_test.shape[0]
    # net_dataidx_map = {}
    # net_dataidx_map_test = {}
    #
    # while min_size < 100:
    #     idx_batch = [[] for _ in range(n_nets)]
    #     idx_batch_test = [[] for _ in range(n_nets)]
    #     for k in range(K):
    #         idx_k = np.where(y_train == k)[0]
    #         idx_k_test = np.where(y_test == k)[0]
    #         np.random.shuffle(idx_k)
    #         np.random.shuffle(idx_k_test)
    #         proportions = np.random.dirichlet(np.repeat(alpha, n_nets))
    #         ## Balance
    #         proportions = np.array([p * (len(idx_j) < N / n_nets) for p, idx_j in zip(proportions, idx_batch)])
    #         proportions_test = np.array(
    #             [p * (len(idx_j) < N_test / n_nets) for p, idx_j in zip(proportions, idx_batch_test)])
    #         proportions = proportions / proportions.sum()
    #         proportions_test = proportions_test / proportions_test.sum()
    #         proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
    #         proportions_test = (np.cumsum(proportions_test) * len(idx_k_test)).astype(int)[:-1]
    #         idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
    #         idx_batch_test = [idx_j + idx.tolist() for idx_j, idx in
    #                           zip(idx_batch_test, np.split(idx_k_test, proportions_test))]
    #         min_size = min([len(idx_j) for idx_j in idx_batch])
    #         min_size_test = min([len(idx_j) for idx_j in idx_batch_test])

    for j in range(n_nets):
        np.random.shuffle(idx_batch[j])
        np.random.shuffle(idx_batch_test[j])
        net_dataidx_map[j] = idx_batch[j]
        net_dataidx_map_test[j] = idx_batch_test[j]

    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)
    testdata_cls_counts = record_net_data_stats(y_test, net_dataidx_map_test)

    return (X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts, net_dataidx_map_test, testdata_cls_counts)

def get_dataloader(dataset, datadir, flag, dataidxs=None):

    if dataset == 'mnist':
        dl_obj = MNIST_truncated
    elif dataset == 'cifar10':
        dl_obj = CIFAR10_truncated
    elif dataset == 'cifar100':
        dl_obj = CIFAR100_truncated

    if flag:
        data = dl_obj(datadir, dataidxs=dataidxs, train=True, download=True)
    else:
        data = dl_obj(datadir, dataidxs=dataidxs, train=False, download=True)

    return data


parser = get_parser()
args = parser.parse_args()

X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts, net_dataidx_map_test, testdata_cls_counts = partition_data(args.dataset, args.datadir, args.total_clients, 0.5)
# X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts= utils.partition_data(args.dataset, args.datadir, args.total_clients, 16)

# traindata_cls_counts:每个client有哪几类，每类有多少
# net_dataidx_map: 有哪些图片
# np.save('net_dataidx_map.npy', net_dataidx_map)

for client_id in range(args.total_clients):

    dataidxs = net_dataidx_map[client_id]
    dataidxs_test = net_dataidx_map_test[client_id]

    # print(dataidxs)
    train_dl = get_dataloader(args.dataset, args.datadir, 1, dataidxs)
    test_dl = get_dataloader(args.dataset, args.datadir, 0, dataidxs_test)

    train_data_x = train_dl.data

    train_data_y = train_dl.target
    train_data = {'x': train_data_x.tolist(), 'y': train_data_y.tolist()}

    test_data_x = test_dl.data
    test_data_y = test_dl.target
    test_data = {'x': test_data_x.tolist(), 'y': test_data_y.tolist()}


    print(client_id)

    #501
    with open('/data/yc/leaf_te_data/cifar10/data/noniid_train/train_'+ str(client_id + 1) + '.json', 'w') as outfile:
        json.dump(train_data, outfile)
    #
    with open('/data/yc/leaf_te_data/cifar10/data/noniid_test/test_'+ str(client_id + 1) + '.json', 'w') as outfile:
        json.dump(test_data, outfile)

    #519
    # with open('/data6T/yangchen/leaf-tes/data/cifar10/data/train/train_'+ str(client_id + 1) + '.json', 'w') as outfile:
    #     json.dump(train_data, outfile)

    # with open('/data6T/yangchen/leaf-tes/data/cifar10/data/test/test_'+ str(client_id + 1) + '.json', 'w') as outfile:
    #     json.dump(test_data, outfile)
