# -*- coding: utf-8 -*
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


DATASETS = ['sent140', 'femnist', 'shakespeare', 'celeba', 'synthetic', 'reddit', 'cifar10']

parser = argparse.ArgumentParser(description='mutate for difference-inducing input generation in cifar10 dataset')
parser.add_argument('-dataset',
                    help='name of dataset;',
                    type=str,
                    choices=DATASETS,
                    default='cifar10')
parser.add_argument('-operator', help="realistic operate type", choices=['light', 'noise', 'translation', 'rotation', 'mask'], default='rotation')

args = parser.parse_args()

user = []
train_num_sample = []
test_num_sample = []
user_data_train = {}
user_data_test = {}
train_file_path = ''
test_file_path = ''

img_rows = 32
img_cols = 32

def image_mask(gen_img, x, y, w, h, size):
    # 马赛克的实现原理是把图像上某个像素点一定范围邻域内的所有点用邻域内左上像素点的颜色代替，这样可以模糊细节，但是可以保留大体的轮廓。
    # x 左顶点 y 右顶点 w 马赛克宽  h 马赛克高 size 马赛克每一块大小
    fh, fw = gen_img.shape[0], gen_img.shape[1]
    if (y + h > fh) or (x + w > fw):
        return
    for i in range(0, h - size, size):  # 关键点0 减去neightbour 防止溢出
        for j in range(0, w - size, size):
            rect = [j + x, i + y, size, size]
            color = gen_img[i + y][j + x].tolist()  # 关键点1 tolist
            left_up = (rect[0], rect[1])
            right_down = (rect[0] + size - 1, rect[1] + size - 1)  # 关键点2 减去一个像素
            cv2.rectangle(gen_img, left_up, right_down, color, -1)
    return gen_img

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

def sp_noise(image, prob):
    '''
    添加椒盐噪声
    prob:噪声比例
    '''
    output = np.zeros(image.shape, np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output

def gasuss_noise(image, mean=0, var=0.001):
    '''
        添加高斯噪声
        mean : 均值
        var : 方差
    '''
    image = np.array(image/255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out * 255)
    return out

def read(flag):

    # local data
    # train_data_dir = os.path.join('..',  'data', 'train')
    # test_data_dir = os.path.join('..', 'data', 'test')

    # 501
    # root_pa = '/data2'
    # 519 A40
    root_pa = '/data'
    dataset = args.dataset
    train_data_dir = os.path.join(root_pa, 'yc', 'leaf_new_data', dataset, 'data', 'notrans_train_64')
    test_data_dir = os.path.join(root_pa, 'yc', 'leaf_new_data', dataset, 'data', 'notrans_test_64')

    # 519 /data6T/yangchen/leaf-tes/data/celeba/data/train
    # root_pa = '/data6T'
    # dataset = args.dataset
    # train_data_dir = os.path.join(root_pa, 'yangchen', 'leaf-tes', 'data', dataset, 'data', 'train')
    # test_data_dir = os.path.join(root_pa, 'yangchen', 'leaf-tes', 'data', dataset, 'data', 'test')

    users, groups, train_data, test_data, train_file_path, test_file_path = read_data(
        train_data_dir, test_data_dir, flag)

    # 添加测试集
    train_x = train_data['x']
    train_y = train_data['y']


    test_x = test_data['x']
    test_y = test_data['y']

    # train_x.append(train_data['x'])
    # train_y.append(train_data['y'])

    # test_x.append(test_data['x'])
    # test_y.append(test_data['y'])

    return train_x, train_y, test_x, test_y, train_file_path, test_file_path

def wrt_json(train_x, train_y, test_x, test_y, train_file_path, test_file_path):
    all_data_train = {}
    all_data_train['x'] = train_x
    all_data_train['y'] = train_y

    all_data_test = {}
    all_data_test['x'] = test_x
    all_data_test['y'] = test_y

    # print(all_data_test)

    train_file_path = train_file_path.replace('train_64', 'train_react_rotation_10_60_64_data', 1)
    test_file_path = test_file_path.replace('test_64', 'test_react_rotation_10_60_64_data', 1)

    with open(train_file_path, 'w') as outfile:
        json.dump(all_data_train, outfile)

    with open(test_file_path, 'w') as outfile:
        json.dump(all_data_test, outfile)

def copy(flag):

    train_x, train_y, test_x, test_y, train_file_path, test_file_path = read(flag)

    wrt_json(train_x, train_y, test_x, test_y, train_file_path, test_file_path)

def mutation():

    # 建立文件夹
    os.makedirs('/data/yc/leaf_new_data/cifar10/data/notrans_train_react_rotation_10_60_64_data')
    os.makedirs('/data/yc/leaf_new_data/cifar10/data/notrans_test_react_rotation_10_60_64_data')

    init_list = list(range(1, 101))
    # init_list = list(range(1, 7))
    # change 代表选择多少个client，5%为3, 10%为5， 20%为10， 30%为15， 40%为20, 60%为30
    # change 代表选择多少个client，5%为5, 10%为10， 20%为20， 30%为30， 40%为40, 60%为60
    choose_list = random.sample(init_list, 10)      #从inst_list选 个
    other_list = init_list.copy()   #copy与init_list一样的

    #other保留未被选中的
    for change in range(len(choose_list)):
        if choose_list[change] in other_list:
            other_list.remove(choose_list[change])

    # print(choose_list)
    for change in range(len(choose_list)):
        print('=============================')
        print('change clients:' + str(change + 1))

        train_x, train_y, test_x, test_y, train_file_path, test_file_path = read(choose_list[change])

        # 每次随机决定旋转和平移的大小
        # ro_params = random.choice([10, 20, 30, 40, 50, 60, 70, 80, 90])
        # light: default 0.98 noise: default 0.1
        # light_params = round(np.random.random()*1/10, 2)
        # noise_params = round(np.random.random()/10, 2)
        light_params = random.choice(list(range(500, 1001, 1))) / 1000
        # noise_params = random.choice(list(range(0, 400, 1))) / 1000
        noise_params = random.choice(list(range(0, 300, 1))) / 1000
        gasuss_noise_var = random.choice(list(range(0, 100, 1))) / 1000 # default 0.02

        ro_params = random.choice(list(range(-60, 61, 1)))

        # tran_param = random.choice([-1, -2, -3, -4, -5, -6, -7, -8, -9, -10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        # trans_param = random.choice([-1, -2, -3, -4, -5, -6, -7, -8, -9, -10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        tran_param = random.choice(list(range(-15, 16, 1)))
        trans_param = random.choice(list(range(-15, 16, 1)))

        tran_params = [tran_param, trans_param]

        mask_x = random.choice(list(range(13, 20, 1)))
        mask_y = random.choice(list(range(18, 25, 1)))
        mask_h = random.choice(list(range(5, 9, 1)))
        mask_w = random.choice(list(range(5, 9, 1)))
        mask_size = random.choice(list(range(2, 5, 1)))

        # print(len(train_x))

        for img_index_train in range(len(train_x)):

            train_x_value2 = train_x[img_index_train]
            train_x_value2 = np.array(train_x_value2)

            # print(train_x_value2.shape)

            # cv2.imwrite('../inputs/mask_init.png', train_x_value2)

            # train_x_value2 = train_x_value2.reshape(1, img_rows, img_cols, 3)

            train_img = train_x_value2.reshape(img_rows, img_cols, 3).astype(np.uint8)
            # cv2.imwrite('../inputs/light_init.png', train_x_value2)

            # 大于1变暗和小于
            if args.operator == 'light':
                # muta_img = exposure.adjust_gamma(gen_img, 5)
                muta_train_img = exposure.adjust_gamma(train_img, light_params)
            elif args.operator == 'noise':
                muta_train_img = sp_noise(train_img, noise_params)
                # muta_train_img = gasuss_noise(train_img, 0, gasuss_noise_var)
            elif args.operator == 'translation':
                muta_train_img = image_translation(train_img, tran_params)
            elif args.operator == 'rotation':
                muta_train_img = image_rotation(train_img, ro_params)
            elif args.operator == 'mask':
                muta_train_img = image_mask(train_img, mask_x, mask_y, mask_h, mask_w, mask_size)

            # muta_train_img = muta_train_img.reshape(img_rows, img_cols, 3)

            # cv2.imwrite('../inputs/mask_mutate.png', muta_train_img)

            muta_train_img = muta_train_img.tolist()

            train_x[img_index_train] = muta_train_img

        for img_index_test in range(len(test_x)):

            test_x_value2 = test_x[img_index_test]
            test_x_value2 = np.array(test_x_value2)

            test_img = test_x_value2.reshape(img_rows, img_cols, 3).astype(np.uint8)

            if args.operator == 'light':
                # muta_img = exposure.adjust_gamma(gen_img, 5)
                muta_test_img = exposure.adjust_gamma(test_img, light_params)
            elif args.operator == 'noise':
                muta_test_img = sp_noise(test_img, noise_params)
                # muta_test_img = gasuss_noise(test_img, 0, gasuss_noise_var)
            elif args.operator == 'translation':
                muta_test_img = image_translation(test_img, tran_params)
            elif args.operator == 'rotation':
                muta_test_img = image_rotation(test_img, ro_params)
            elif args.operator == 'mask':
                muta_test_img = image_mask(test_img, mask_x, mask_y, mask_h, mask_w, mask_size)

            muta_test_img = muta_test_img.tolist()

            test_x[img_index_test] = muta_test_img

        wrt_json(train_x, train_y, test_x, test_y, train_file_path, test_file_path)

    # with open('/data2/yc/leaf_te_data/cifar10/data/notrans_train_react_noise_40_0.4_64_new/setup.txt', "w",
    with open('/data/yc/leaf_new_data/cifar10/data/notrans_train_react_rotation_10_60_64_data/setup.txt', "w",
              encoding='utf-8') as fw:
        j = 0
        for word in choose_list:
            if j != len(choose_list) - 1:
                fw.write(str(word) + ',')
            else:
                fw.write(str(word))
            j += 1

    for copy_count in range(len(other_list)):
        print('=============================')
        print('copy clients:' + str(copy_count + 1))
        copy(other_list[copy_count])

mutation()
