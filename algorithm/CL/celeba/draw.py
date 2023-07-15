# -*-coding:utf-8-*-
# 折线图

import matplotlib.pyplot as plt
import numpy as np
import pylab as pl

flag1 = 'accuracy'
flag2 = 'loss'


fonten = {'family': 'Times New Roman', 'size': '13'}
table_x_label = 'Epoch'
table_y_label = "Accuracy"


Train_Acc = []
Test_Acc = []
train_acc = open('./loss/new_train_acc.txt', "r", encoding='utf-8')
test_acc = open('./loss/new_test_acc.txt', "r", encoding='utf-8')
# train_acc = open('../result/femnist/MLP3/normal/test_0.01/new_train_accuracy.txt', "r", encoding='utf-8')
# test_acc = open('../result/femnist/MLP3/normal/test_0.01/new_test_accuracy.txt', "r", encoding='utf-8')

for line in train_acc.readlines():
    line = line.strip()
    line = line.strip('\n')
    line = line.split(',')
    # Acc.append(line)
    for word in line:
        Train_Acc.append(float(word)/100.)
        # print(len(Train_Acc))

for line in test_acc.readlines():
    line = line.strip()
    line = line.strip('\n')
    line = line.split(',')
    # Acc.append(line)
    for word in line:
        Test_Acc.append(float(word)/100.)
        # print(len(Test_Acc))

time = []

for i in range(len(Test_Acc)):
    time.append(i)



plt.figure()

Train_Acc = np.array(Train_Acc)
Test_Acc = np.array(Test_Acc)
time = np.array(time)


# plt.figure()
#
# plt.plot(epochs, loss, 'bo', label='Training loss')
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.legend()

# plt.show()


fig, ax = plt.subplots()
#
ax.plot(time, Train_Acc, '-', color='r', label='Train_Accuracy')
ax.plot(time, Test_Acc, '-', color='b', label='Test_Accuracy')
#
# plt.plot(time, Acc, 'g.-', alpha=0.8, color='green', linewidth=2)

# plt.tick_params(labelsize=15)

plt.xlabel(table_x_label, fontdict=fonten, fontsize=15)
plt.ylabel(table_y_label, fontdict=fonten, fontsize=15)
#
# # 纵轴范围区间

#celeba acc
# my_y_ticks = np.arange(0.7, 0.96, 0.05)
# my_x_ticks = np.arange(0, 1201, 100)

#celeba loss
# my_y_ticks = np.arange(0.15, 0.46, 0.05)
# my_x_ticks = np.arange(0, 1201, 100)


#femnist acc
my_y_ticks = np.arange(0.4, 1.01, 0.05)
# my_y_ticks = np.arange(0, 1, 0.1)
my_x_ticks = np.arange(0, 201, 20)

#cifar loss
# my_y_ticks = np.arange(0, 1.8, 0.2)
# my_x_ticks = np.arange(0, 201, 20)

# my_x_ticks = np.linspace(0, 700, 10)
# my_y_ticks = np.linspace(0, 1, 10)
# plt.yticks(np.linspace(0.45, 0.96, num=10, endpoint=False))
# plt.xticks(np.linspace(0, 701, num=35, endpoint=False))
plt.yticks(my_y_ticks)
plt.xticks(my_x_ticks)
# plt.ylim(0, 1)
# plt.xlim(0, 701)
plt.legend(loc='best')
plt.grid(linestyle='-.')
# plt.grid(True) ##增加格点
#
# plt.axis('tight')
# plt.title('Loss', fontdict=fonten, fontsize=15)
# plt.title('Accuracy', fontdict=fonten, fontsize=15)
plt.title('0.03+T_max200_Accuracy', fontdict=fonten, fontsize=15)
# plt.legend(loc='lower right')
# plt.legend(loc='upper right')
# plt.legend(prop=fonten)
# plt.savefig("../result/femnist/pic/MLP3/normal_loss.png", dpi=800, bbox_inches='tight')
plt.savefig("./result/0.03+T_max200_Accuracy.png", dpi=800, bbox_inches='tight')
# plt.savefig("../result/celeba/pic/L2_fixed_0.1.png")