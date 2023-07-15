import numpy as np
from keras import backend as K
from keras.layers import Input
import argparse
from model_utils import read_data
from operate_utils import *
import random
import cv2
import json
# from scipy.misc import imsave

import os

DATASETS = ['sent140', 'femnist', 'shakespeare', 'celeba', 'synthetic', 'reddit']

parser = argparse.ArgumentParser(description='Main function for difference-inducing input generation in FEMNIST dataset')
# parser.add_argument('grad_iterations', help="number of iterations of gradient descent", type=int, default=20)
parser.add_argument('-dataset',
                    help='name of dataset;',
                    type=str,
                    choices=DATASETS,
                    default='celeba')

parser.add_argument('-operate', help="realistic transformation type", choices=['translation', 'rotation'], default='rotation')

args = parser.parse_args()

user = []
train_num_sample = []
test_num_sample = []
user_data_train = {}
user_data_test = {}
train_file_path = ''
test_file_path = ''


def read():

    # local data
    # train_data_dir = os.path.join('..',  'data', 'train')
    # test_data_dir = os.path.join('..', 'data', 'test')

    root_pa = '/data'
    dataset = args.dataset
    train_data_dir = os.path.join(root_pa, 'yc', 'leaf_te_data', dataset, 'data', 'train')
    test_data_dir = os.path.join(root_pa, 'yc', 'leaf_te_data', dataset, 'data', 'test')

    users, groups, train_data, test_data, train_num_sample, test_num_sample, train_file_path, test_file_path = read_data(train_data_dir, test_data_dir, 1)

    #添加测试集
    train_all_x = []

    test_all_x = []

    user = list(test_data.keys())

    for u in user:

        train_all_x.append(train_data[u]['x'])
        test_all_x.append(test_data[u]['x'])

    return train_all_x, test_all_x

def image_translation(img, params):
    # rows, cols, ch = img.shape
    rows, cols, ch = img.shape

    M = np.float32([[1, 0, params[0]], [0, 1, params[1]]])
    dst = cv2.warpAffine(img, M, (cols, rows))
    return dst

def image_rotation(img, params):
    # rows, cols, ch = img.shape
    rows, cols, ch = img.shape
    M = cv2.getRotationMatrix2D((cols/2, rows/2), params, 1)
    dst = cv2.warpAffine(img, M, (cols, rows))
    return dst

def mutation(dataset, train_all_x, test_all_x):

    train_all_x = np.array(train_all_x)
    test_all_x = np.array(test_all_x)

    # change 代表选择多少个client，5%为19, 10%为38， 20%为76 femnist:379
    # change 代表选择多少个client，5%为42, 10%为84， 20%为168 celeba:838
    for change in range(168):

        index1 = random.randrange(len(test_all_x))

        train_x_value1 = train_all_x[index1]
        test_x_value1 = test_all_x[index1]

        train_x_value1 = np.array(train_x_value1)
        test_x_value1 = np.array(test_x_value1)
        # print(len(test_all_x))

        #每次随机决定旋转和平移的大小
        ro_params = random.choice([10, 20, 30, 40, 50, 60, 70, 80, 90])

        tran_param = random.choice([-4, -8, -12, -16, -20, -24, -28, -32, -36, -40, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40])
        trans_param = random.choice([-4, -8, -12, -16, -20, -24, -28, -32, -36, -40, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40])
        tran_params = [tran_param, trans_param]

        root_pa = '/data'

        for img_index in range(len(train_x_value1)):

            train_x_value2 = train_x_value1[img_index]

            # local data
            # gen_img_dir = os.path.join('..', 'data', 'raw', 'img_align_celeba', train_x_value2)

            gen_img_dir = os.path.join(root_pa, 'yc', 'leaf_te_data', dataset, 'data', 'raw', 'img_align_celeba_rotation_20', train_x_value2)

            # gen_img_deprocessed.shape(218,178,3)
            gen_img_deprocessed = cv2.imread(gen_img_dir)
            # cv2.imwrite('../data/inputs/rotation_init.png', gen_img_deprocessed)

            # imsave('../inputs/translation_init.png', gen_img_deprocessed)

            if args.operate == 'rotation':
                # params为旋转角度
                # ro_params = 60
                muta_img_deprocessed = image_rotation(gen_img_deprocessed, ro_params)
            elif args.operate == 'translation':
                # params为平移距离
                # tran_params = [10, 10]
                muta_img_deprocessed = image_translation(gen_img_deprocessed, tran_params)

            # imsave('../inputs/translation_' + str(tran_param) + '.png', muta_img_deprocessed)
            # cv2.imwrite('../data/inputs/rotation_init' + str(tran_param) + '.png', muta_img_deprocessed)

            #gpu data
            muta_img_dir = os.path.join(root_pa, 'yc', 'leaf_te_data', dataset, 'data', 'raw', 'img_align_celeba_rotation_20', train_x_value2)

            #在突变图片目录把突变后的图片覆盖原图片
            cv2.imwrite(muta_img_dir, muta_img_deprocessed)

        for img_index_1 in range(len(test_x_value1)):

            test_x_value2 = test_x_value1[img_index_1]

            gen_img_dir = os.path.join(root_pa, 'yc', 'leaf_te_data', dataset, 'data', 'raw', 'img_align_celeba_rotation_20', test_x_value2)

            gen_img_deprocessed = cv2.imread(gen_img_dir)

            if args.operate == 'rotation':
                # params为旋转角度
                # ro_params = 60
                muta_img_deprocessed = image_rotation(gen_img_deprocessed, ro_params)
            elif args.operate == 'translation':
                # params为平移距离
                # tran_params = [10, 10]
                muta_img_deprocessed = image_translation(gen_img_deprocessed, tran_params)

            #将转好的图片重新归一化为0,1tensor
                muta_img_dir = os.path.join(root_pa, 'yc', 'leaf_te_data', dataset, 'data', 'raw', 'img_align_celeba_rotation_20', test_x_value2)

            # 在突变图片目录把突变后的图片覆盖原图片
            cv2.imwrite(muta_img_dir, muta_img_deprocessed)

train_all_x, test_all_x = read()
mutation(args.dataset, train_all_x, test_all_x)

