import json
import os
import argparse
import numpy as np
import random
from model_utils import read_data
from skimage import exposure
# from scipy.misc import imsave
from PIL import Image
import cv2


DATASETS = ['sent140', 'femnist', 'shakespeare', 'celeba', 'synthetic', 'reddit']

parser = argparse.ArgumentParser(description='mutate for difference-inducing input generation in FEMNIST dataset')
parser.add_argument('-dataset',
                    help='name of dataset;',
                    type=str,
                    choices=DATASETS,
                    default='celeba')
parser.add_argument('-operator', help="realistic operate type", choices=['light', 'noise', 'translation', 'rotation', 'mask'], default='light')

args = parser.parse_args()

user = []
train_num_sample = []
test_num_sample = []
user_data_train = {}
user_data_test = {}
train_file_path = ''
test_file_path = ''

img_rows = 84
img_cols = 84

def read():

    # local data
    train_data_dir = os.path.join('..',  'data', 'train')
    test_data_dir = os.path.join('..', 'data', 'test')

    # 501
    # root_pa = '/data2'
    # dataset = args.dataset
    # train_data_dir = os.path.join(root_pa, 'yc', 'leaf_te_data', dataset, 'data', 'train')
    # test_data_dir = os.path.join(root_pa, 'yc', 'leaf_te_data', dataset, 'data', 'test')

    # 519 /data6T/yangchen/leaf-tes/data/celeba/data/train
    # root_pa = '/data6T'
    # dataset = args.dataset
    # train_data_dir = os.path.join(root_pa, 'yangchen', 'leaf-tes', 'data', dataset, 'data', 'train')
    # test_data_dir = os.path.join(root_pa, 'yangchen', 'leaf-tes', 'data', dataset, 'data', 'test')

    users, groups, train_data, test_data, train_num_sample, test_num_sample, train_file_path, test_file_path = read_data(
        train_data_dir, test_data_dir, 1)

    # 添加测试集
    train_all_x = []

    test_all_x = []

    user = list(test_data.keys())

    for u in user:
        train_all_x.append(train_data[u]['x'])
        test_all_x.append(test_data[u]['x'])

    return train_all_x, test_all_x


def new_copy( train_all_x, test_all_x):
    t_dir='train_need_data'
    te_dir='test_need_data'
    os.makedirs('/data/yc/leaf_new_data/celeba/data/'+t_dir)
    os.makedirs('/data/yc/leaf_new_data/celeba/data/'+te_dir)


    train_all_x = np.array(train_all_x)
    test_all_x = np.array(test_all_x)

    for copy_1 in range(len(train_all_x)):
        train_x_value1 = train_all_x[copy_1]
        test_x_value1 = test_all_x[copy_1]

        train_x_value1 = np.array(train_x_value1)
        test_x_value1 = np.array(test_x_value1)

        for copy_1_train in range(len(train_x_value1)):
            train_x_value2 = train_x_value1[copy_1_train]


            gen_img_dir = os.path.join( '/data','yc','leaf_new_data','celeba','data', 'raw', 'img_align_celeba', train_x_value2)


            gen_img = cv2.imread(gen_img_dir)

            muta_img_dir = os.path.join('/data','yc','leaf_new_data','celeba','data',  t_dir, train_x_value2)

            cv2.imwrite(muta_img_dir, gen_img)

        for copy_1_test in range(len(test_x_value1)):
            test_x_value2 = test_x_value1[copy_1_test]

            gen_img_dir = os.path.join('/data', 'yc', 'leaf_new_data', 'celeba', 'data', 'raw', 'img_align_celeba',
                                       test_x_value2)

            gen_img = cv2.imread(gen_img_dir)

            muta_img_dir = os.path.join('/data', 'yc', 'leaf_new_data', 'celeba', 'data', te_dir, test_x_value2)

            cv2.imwrite(muta_img_dir, gen_img)




train_all_x, test_all_x = read()

new_copy(train_all_x, test_all_x)