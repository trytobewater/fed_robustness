import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
import logging
import argparse
import numpy as np
import time
import torch
import torch.nn as nn
from datetime import datetime
from torchvision import datasets, transforms

from log import setup_logging, ResultsLog, save_checkpoint
from meters import AverageMeter, accuracy
from preprocess import get_transform, get_int8_transform
from lenet import lenet
from lenet import lenet_celeba
from mobilenet import mobilenet
import utils
import cifar_data_loader

from vgg import VGG_cifar
from vgg import VGG_celeba
from optim import OptimRegime
from data import get_dataset
from ti_lenet import TiLenet
from ti_vgg import TiVGG_cifar
import torch.optim as optim
import torch.nn.functional as F
import ti_torch
from torch.utils.tensorboard import SummaryWriter
import shutil

from client import Client
from server import Server
from args import parse_args

from models.model_utils import read_cifar_data


def main():
    if not torch.cuda.is_available():
        print('Require nvidia gpu with tensor core to run')
        return

    global args
    args = parse_args()

    if args.seed > 0:
        torch.manual_seed(args.seed)
        logging.info("random seed: %s", args.seed)
    else:
        logging.info("random seed: None")

    logging.info("act rounding scheme: %s", ti_torch.ACT_ROUND_METHOD.__name__)
    logging.info("err rounding scheme: %s", ti_torch.ERROR_ROUND_METHOD.__name__)
    logging.info("gradient rounding scheme: %s", ti_torch.GRAD_ROUND_METHOD.__name__)
    if args.weight_frac:
        ti_torch.UPDATE_WITH_FRAC = True
        logging.info("Update WITH Fraction")
    else:
        ti_torch.UPDATE_WITH_FRAC = False

    if args.weight_decay:
        ti_torch.WEIGHT_DECAY = True
        logging.info("Update WITH WEIGHT DECAY")
    else:
        ti_torch.WEIGHT_DECAY = False

    data_dir_modify_here_1 = 'notrans_train_react_noise_10_0.1_64_data'

    data_dir_modify_here_2 = 'notrans_test_react_noise_10_0.1_64_data'

    train_data_dir = os.path.join('./', 'data', 'cifar10', 'data', data_dir_modify_here_1)
    test_data_dir = os.path.join('./', 'data', 'cifar10', 'data', data_dir_modify_here_2)

    users, groups, train_data, test_data = read_cifar_data(train_data_dir, test_data_dir)

    clients = setup_clients(args.model_type, train_data, test_data)

    server = init_server(args.model_type)

    plot_result = []

    for epoch in range(args.start_epoch, args.num_round):

        server.select_clients(epoch, clients, num_clients=args.num_clients)

        server.train_model(epoch, args.num_epochs, args.batch_size, server.selected_clients, args.model_type)

        server.update_model(clients, args.model_type, epoch)

        if epoch % args.log_interval == 0:

            train_metrics = server.test_model(clients, args.model_type, set_to_use='train')

            test_metrics = server.test_model(clients, args.model_type, set_to_use='test')

            train_metrics_list = list(train_metrics.values())
            test_metrics_list = list(test_metrics.values())

            train_avg_acc = 0.
            test_avg_acc = 0.
            train_total_weight = 0
            test_total_weight = 0

            for metric in train_metrics_list:
                train_total_weight += metric.count
                train_avg_acc += metric.count * metric.avg

            for metric in test_metrics_list:
                test_total_weight += metric.count
                test_avg_acc += metric.count * metric.avg
            train_avg_acc = train_avg_acc / train_total_weight
            test_avg_acc = test_avg_acc / test_total_weight

            logging.info('current round {:d} '
                         'train_accuracy {:.4f} '
                         'test_accuracy {:.4f}'
                         .format(epoch, train_avg_acc, test_avg_acc))

            plot_result.append((epoch, train_avg_acc, test_avg_acc))

    np.save('result-niti-float-test-no-nor-base-sgd-cifar10-2.npy', plot_result)


def init_server(model_type='int'):
    if model_type == 'int':
        int_model, _ = generate_model(model_type)
        server = Server(int_model, None)
    elif model_type == 'float':
        float_model, _ = generate_model(model_type)
        server = Server(None, float_model)
    else:
        int_model, _ = generate_model('int')
        float_model, _ = generate_model('float')
        server = Server(int_model, float_model)
    return server


def generate_model(model_type='int'):
    if model_type == 'int':
        logging.info('Create integer model')
        optimizer = None
        if args.dataset == 'mnist':
            model = TiLenet()
        elif args.dataset == 'cifar10':
            if args.model == 'vgg':
                model = TiVGG_cifar(args.depth, 10)

        if args.weight_frac:
            regime = model.regime_frac
        else:
            regime = model.regime
    else:
        if args.dataset == 'mnist' and args.model == 'lenet':
            model = lenet().to('cuda:0')

        elif args.dataset == 'celeba':
            if args.model == 'lenet':
                model = lenet_celeba().to('cuda:0')
            elif args.model == 'vgg':
                model = VGG_celeba(84 * 84, 2).to('cuda:0')

        elif args.dataset == 'cifar10':
            if args.model == 'vgg':
                model = VGG_cifar(args.depth, 10).to('cuda:0')

        elif args.dataset == 'cifar100':
            if args.model == 'mobilenet':
                model = mobilenet(1, 100).to('cuda:0')

        num_parameters = sum([l.nelement() for l in model.parameters()])
        logging.info("created float network on %s", args.dataset)
        logging.info("number of parameters: %d", num_parameters)
        regime = getattr(model, 'regime')
        optimizer = OptimRegime(model.parameters(), regime)
    return model, optimizer


def setup_clients(model_type='int', train_data_loader=None, test_data_loader=None):
    users = [str(i) for i in range(100)]
    clients = []
    model, optimizer = generate_model(model_type)
    for i in range(len(users)):
        u = users[i]

        train_data = train_data_loader['' + str(i + 1) + '']

        test_data = test_data_loader['' + str(i + 1) + '']
        client = Client(u, train_data, test_data, model, optimizer, model_type)
        clients.append(client)

    return clients


def create_clients(users, train_data, test_data, model, optimizer, model_type):
    clients = [Client(u, train_data[u], test_data[u], model, optimizer, model_type) for u in users]
    return clients


def create_hybrid_clients(users, train_data, test_data, model_types):
    clients = []
    model_type_str = ''
    for i, model_type in enumerate(model_types):
        if model_type == 1:
            model_type_str = 'int'
        else:
            model_type_str = 'float'
        model, optimizer = generate_model(model_type_str)
        clients.append(Client(users[i], train_data[users[i]], test_data[users[i]], model, optimizer, model_type_str))
    return clients


if __name__ == '__main__':
    main()
