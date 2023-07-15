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

from models.model_utils import read_cifar_data
from models.model_utils import cifar10_loaders
from models.model_utils import cifar10_singlebatch_loaders

os.environ["CUDA_VISIBLE_DEVICES"] = "6"

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')#0.1
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
# transform_train = transforms.Compose([
#     transforms.RandomCrop(32, padding=4),#依据跟定的size，从中心进行裁剪
#     transforms.RandomHorizontalFlip(),#依据概率p对PIL图片进行水平翻转
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),#对数据按通道进行标准化，即先减均值，再除以标准差
# ])
#
# transform_test = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ])
#
# trainset = torchvision.datasets.CIFAR10(
#     root='/data/yc/leaf_te_data/cifar10/data/raw_data', train=True, download=True, transform=transform_train)
# trainloader = torch.utils.data.DataLoader(
#     trainset, batch_size=128, shuffle=True, num_workers=2)

# testset = torchvision.datasets.CIFAR10(
#     root='/data/yc/leaf_te_data/cifar10/data/raw_data', train=False, download=True, transform=transform_test)
# testloader = torch.utils.data.DataLoader(
#     testset, batch_size=100, shuffle=False, num_workers=2)

# 519 A40
# root_pa = '/data'
#501
root_pa = '/data'
# root_pa = '/data'
train_data_dir = os.path.join(root_pa, 'yc', 'leaf_new_data', 'cifar10', 'data', 'notrans_train_64')#notrans_train_react_noise_60_0.6_64
test_data_dir = os.path.join(root_pa, 'yc', 'leaf_new_data', 'cifar10', 'data', 'notrans_test_64')#torch_test_react_noise_20_0.3_64
#
users, groups, train_data, test_data = read_cifar_data(train_data_dir, test_data_dir)

train_da = []
test_da = []
for u in users:
# #     # print(u)
    train_da.append(train_data[u])
    test_da.append(test_data[u])

# for i in range(len(test_da)):
#     if i == 0:
#         data = np.array(test_da[i]['x'])
#         target = np.array(test_da[i]['y'])
#     else:
#         data = np.concatenate((data, np.array(test_da[i]['x'])), axis=0)
#         target = np.concatenate((target, np.array(test_da[i]['y'])), axis=0)

# #
# trainloader = cifar10_loaders('train', train_da)
# trainloader = cifar10_singlebatch_loaders('train', train_da)
testloader = cifar10_loaders('test', test_da)

# for i in range(100):
#     train_x = np.load('/data/yc/leaf_te_data/cifar10/data/torch_train_init/train_datax_' + str(i + 1) + '.npy', allow_pickle=True)
#     train_y = np.load('/data/yc/leaf_te_data/cifar10/data/torch_train_init/train_datax_' + str(i + 1) + '.npy', allow_pickle=True)
#     test_x = np.load('/data/yc/leaf_te_data/cifar10/data/torch_test_init/test_datay_' + str(i + 1) + '.npy', allow_pickle=True)
#     test_y = np.load('/data/yc/leaf_te_data/cifar10/data/torch_test_init/test_datay_' + str(i + 1) + '.npy', allow_pickle=True)
#     trainloader, testloader = cifar10_loaders(i, train_x, train_y, test_x, test_y)

# import torch.utils.data as Data
# import numpy as np
#
# train_ds = np.load('/data/yc/leaf_te_data/cifar10/data/torch_train_init/train_data.npy', allow_pickle=True).item()
# test_ds = np.load('/data/yc/leaf_te_data/cifar10/data/torch_test_init/test_data.npy', allow_pickle=True).item()
# train_bs = 64
# test_bs = 64
#
# trainloader = Data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True)
# testloader = Data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
net = VGG('VGG16')
# net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
# net = SimpleDLA()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

# if args.resume:
#     # Load checkpoint.
#     print('==> Resuming from checkpoint..')
#     assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
#     checkpoint = torch.load('/data/yc/pytorch-cifar-master/checkpoint/normal_ckpt.pth')
#     net.load_state_dict(checkpoint['net'])
#     best_acc = checkpoint['acc']
#     start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)# Momentum 梯度下降法，就是计算了梯度的指数加权平均数，并以此来更新权重,动量方法相当于把纸团换成了铁球；不容易受到外力的干扰，轨迹更加稳定
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200) #余弦退火学习率,让学习率随epoch的变化图类似于cos，更新策略,Tmax表示cos周期的1/2


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    step_samples_x = np.load('/data/yc/niti_new/centralized_training/cifar10/sample/epoch_' + str(epoch) + '_x.npy')
    step_samples_y = np.load('/data/yc/niti_new/centralized_training/cifar10/sample/epoch_' + str(epoch) + '_y.npy')

    # 从numpy 变成 tensor
    step_samples_x = torch.tensor(step_samples_x)
    step_samples_y = torch.tensor(step_samples_y)

    inputs, targets = step_samples_x.to(device), step_samples_y.to(device)

    optimizer.zero_grad()
    outputs = net(inputs)
    # print(inputs.size()) # torch.Size([128, 3, 32, 32])
    # print(outputs.size())
    # print(targets.size())
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

    train_loss += loss.item()
    _, predicted = outputs.max(1)
    total += targets.size(0)
    correct += predicted.eq(targets).sum().item()

    print('Train Loss: %.3f | Train Acc: %.3f' % (train_loss, 100.*correct/total))


    # step_samples_x = np.load('/data2/yc/FloatFL-cifar10-1/niti-cifar10-float/sample/normal_640/batchSGD_data/epoch_' + str(epoch) + '_x.npy')
    # step_samples_y = np.load('/data2/yc/FloatFL-cifar10-1/niti-cifar10-float/sample/normal_640/batchSGD_data/epoch_' + str(epoch) + '_y.npy')
    #
    # trainloader = cifar10_singlebatch_loaders('train', step_samples_x, step_samples_y)
    #
    # for batch_idx, (inputs, targets) in enumerate(trainloader):
    #
    #     # 保存trainloader中第一个batch的数据
    #     # if epoch == 0 and batch_idx == 0:
    #     #     save_file_x = './sample/epoch_' + str(epoch) + '_x.npy'
    #     #     save_file_y = './sample/epoch_' + str(epoch) + '_y.npy'
    #     #     np.save(save_file_x, inputs)
    #     #     np.save(save_file_y, targets)
    #
    #     inputs, targets = inputs.to(device), targets.to(device)
    #     optimizer.zero_grad()
    #     outputs = net(inputs)
    #     # print(inputs.size()) # torch.Size([128, 3, 32, 32])
    #     # print(outputs.size())
    #     # print(targets.size())
    #     loss = criterion(outputs, targets)
    #     loss.backward()
    #     optimizer.step()
    #
    #     train_loss += loss.item()
    #     _, predicted = outputs.max(1)
    #     total += targets.size(0)
    #     correct += predicted.eq(targets).sum().item()
    #
    #     # print(batch_idx)
    #
    #     progress_bar(batch_idx, len(trainloader), 'Train Loss: %.3f | Train Acc: %.3f%% (%d/%d)'
    #                  % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

        # print(train_loss)
        # print(train_loss/(batch_idx+1))

    # Train_Loss.append(train_loss / (391))
    Train_Loss.append(train_loss)
    Train_acc.append(100.*correct/total)


    # for batch_idx, (inputs, targets) in enumerate(trainloader):
    #
    #     # 保存trainloader中第一个batch的数据
    #     # if epoch == 0 and batch_idx == 0:
    #     #     save_file_x = './sample/epoch_' + str(epoch) + '_x.npy'
    #     #     save_file_y = './sample/epoch_' + str(epoch) + '_y.npy'
    #     #     np.save(save_file_x, inputs)
    #     #     np.save(save_file_y, targets)
    #
    #     inputs, targets = inputs.to(device), targets.to(device)
    #     optimizer.zero_grad()
    #     outputs = net(inputs)
    #     # print(inputs.size()) # torch.Size([128, 3, 32, 32])
    #     # print(outputs.size())
    #     # print(targets.size())
    #     loss = criterion(outputs, targets)
    #     loss.backward()
    #     optimizer.step()
    #
    #     train_loss += loss.item()
    #     _, predicted = outputs.max(1)
    #     total += targets.size(0)
    #     correct += predicted.eq(targets).sum().item()
    #
    #     # print(batch_idx)
    #
    #     progress_bar(batch_idx, len(trainloader), 'Train Loss: %.3f | Train Acc: %.3f%% (%d/%d)'
    #                  % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    #
    #     # print(train_loss)
    #     # print(train_loss/(batch_idx+1))
    #
    # Train_Loss.append(train_loss / (391))
    # Train_acc.append(100.*correct/total)





def test(epoch):
# def test(epoch, data, target):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():

        #每79/40个epoch测试一次, 50000/640 = 78.125, 50000/1280 = 39.0625
        # if epoch % 79 == 0:
        #     inputs = torch.tensor(data)
        #     targets = torch.tensor(target)
        #
        #     inputs, targets = inputs.to(device), targets.to(device)
        #     outputs = net(inputs)
        #
        #     loss = criterion(outputs, targets)
        #
        #     test_loss += loss.item()
        #     _, predicted = outputs.max(1)
        #     total += targets.size(0)
        #     correct += predicted.eq(targets).sum().item()
        #
        #     print('Test Loss: %.3f | Test Acc: %.3f' % (test_loss, 100. * correct / total))

        # 每79/40个epoch测试一次, 50000/640 = 78.125, 50000/1280 = 39.0625
        if epoch % 79 == 0 or epoch == 14999:
        # if epoch % 40 == 0 or epoch == 7499:

            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)

                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()


                progress_bar(batch_idx, len(testloader), 'Test Loss: %.3f | Test Acc: %.3f%% (%d/%d)'
                             % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

            Test_Loss.append(test_loss / (100))
            Test_acc.append(100. * correct / total)

#     # Save checkpoint.
#     acc = 100.*correct/total
#     if acc > best_acc:
#         print('Saving..')
#         state = {
#             'net': net.state_dict(),
#             'acc': acc,
#             'epoch': epoch,
#         }
#         if not os.path.isdir('checkpoint'):
#             os.mkdir('checkpoint')
#         torch.save(state, '/data/yc/pytorch-cifar-master/checkpoint/normal_ckpt.pth')
#         best_acc = acc
#
Train_Loss = []
Train_acc = []
Test_Loss = []
Test_acc = []

# for epoch in range(start_epoch, start_epoch+200):
for epoch in range(15000):
# for epoch in range(7500):
    train(epoch)
    test(epoch)
    # test(epoch, data, target)
    # scheduler.step()


with open('/data2/yc/pytorch-cifar-master/result/batchSGD_acc/train_noise60_0.6_random_640_loss.txt', "w", encoding='utf-8') as fw:
    j = 0
    for word in Train_Loss:
        if j != len(Train_Loss) - 1:
            fw.write(str(word) + ',')
        else:
            fw.write(str(word))
        j += 1

with open('/data2/yc/pytorch-cifar-master/result/batchSGD_acc/test_noise60_0.6_random_640_loss.txt', "w", encoding='utf-8') as fw:
    j = 0
    for word in Test_Loss:
        if j != len(Test_Loss) - 1:
            fw.write(str(word) + ',')
        else:
            fw.write(str(word))
        j += 1

with open('/data2/yc/pytorch-cifar-master/result/batchSGD_acc/train_noise60_0.6_random_640_acc.txt', "w", encoding='utf-8') as fw:
    j = 0
    for word in Train_acc:
        if j != len(Train_acc) - 1:
            fw.write(str(word) + ',')
        else:
            fw.write(str(word))
        j += 1

with open('/data2/yc/pytorch-cifar-master/result/batchSGD_acc/test_noise60_0.6_random_640_acc.txt', "w", encoding='utf-8') as fw:
    j = 0
    for word in Test_acc:
        if j != len(Test_acc) - 1:
            fw.write(str(word) + ',')
        else:
            fw.write(str(word))
        j += 1

#notran_nosche_new_noise_40_0.4_64_test_acc