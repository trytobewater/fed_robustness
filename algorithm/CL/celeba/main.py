'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar
import os
import numpy as np

from models.model_utils import read_data
from models.model_utils import cifar10_loaders
from models.lenet5 import lenet_celeba

os.environ["CUDA_VISIBLE_DEVICES"] = "6"

parser = argparse.ArgumentParser(description='PyTorch celeba Training')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0
start_epoch = 0

print('==> Preparing data..')

root_pa = '/data'
train_data_dir = os.path.join(root_pa, 'yc', 'leaf_new_data', 'celeba', 'data', 'train')
test_data_dir = os.path.join(root_pa, 'yc', 'leaf_new_data', 'celeba', 'data', 'test')

users, groups, train_data, test_data = read_data(train_data_dir, test_data_dir)

train_da = []
test_da = []
for u in users:
    train_da.append(train_data[u])
    test_da.append(test_data[u])

trainloader = cifar10_loaders('train', train_da)
testloader = cifar10_loaders('test', test_da)

print('==> Building model..')
net = lenet_celeba()

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('/data/yc/niti_new/centralized_training/cifar10/checkpoint/normal_ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)


def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):

        if epoch == 0 and batch_idx == 0:
            save_file_x = './sample/epoch_' + str(epoch) + '_x.npy'
            save_file_y = './sample/epoch_' + str(epoch) + '_y.npy'
            np.save(save_file_x, inputs)
            np.save(save_file_y, targets)

        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Train Loss: %.3f | Train Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    Train_Loss.append(train_loss / (391))
    Train_acc.append(100. * correct / total)


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)

            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Test Loss: %.3f | Test Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

        Test_Loss.append(test_loss / (100))
        Test_acc.append(100. * correct / total)

    acc = 100. * correct / total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')

        best_acc = acc


Train_Loss = []
Train_acc = []
Test_Loss = []
Test_acc = []

for epoch in range(start_epoch, start_epoch + 200):
    train(epoch)
    test(epoch)

print(max(Test_acc))
