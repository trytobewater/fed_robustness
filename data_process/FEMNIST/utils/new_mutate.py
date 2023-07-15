import json
import os
import argparse
import numpy as np
import random
from model_utils import read_data
from skimage import exposure
# from scipy.misc import imsave
import cv2


DATASETS = ['sent140', 'femnist', 'shakespeare', 'celeba', 'synthetic', 'reddit']

parser = argparse.ArgumentParser(description='mutate for difference-inducing input generation in FEMNIST dataset')
parser.add_argument('-dataset',
                    help='name of dataset;',
                    type=str,
                    choices=DATASETS,
                    default='femnist')
parser.add_argument('-operator', help="realistic operate type", choices=['light', 'noise', 'translation', 'rotation'], default='translation')

args = parser.parse_args()

user = []
train_num_sample = []
test_num_sample = []
user_data_train = {}
user_data_test = {}
train_file_path = ''
test_file_path = ''

img_rows = 28
img_cols = 28

def deprocess_image(x):
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x.reshape(x.shape[1], x.shape[2])

def image_translation(img, params):
    # rows, cols, ch = img.shape
    rows, cols = img.shape

    M = np.float32([[1, 0, params[0]], [0, 1, params[1]]])
    dst = cv2.warpAffine(img, M, (cols, rows), borderValue=(255, 255))
    return dst

def image_rotation(img, params):
    # rows, cols, ch = img.shape
    rows, cols = img.shape
    M = cv2.getRotationMatrix2D((cols/2, rows/2), params, 1)
    dst = cv2.warpAffine(img, M, (cols, rows), borderValue=(255, 255))
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

def read():

    # local data
    # train_data_dir = os.path.join('..',  'data', 'train')
    # test_data_dir = os.path.join('..', 'data', 'test')

    root_pa = '/data'
    dataset = args.dataset
    train_data_dir = os.path.join(root_pa, 'yc', 'leaf_new_data', dataset, 'data', 'train')
    test_data_dir = os.path.join(root_pa, 'yc', 'leaf_new_data', dataset, 'data', 'test')

    users, groups, train_data, test_data, train_num_sample, test_num_sample, train_file_path, test_file_path = read_data(
        train_data_dir, test_data_dir, 1)

    # 添加测试集
    train_all_x = []
    train_all_y = []

    test_all_x = []
    test_all_y = []

    user = list(test_data.keys())

    for u in user:

        train_all_x.append(train_data[u]['x'])
        train_all_y.append(train_data[u]['y'])

        test_all_x.append(test_data[u]['x'])
        test_all_y.append(test_data[u]['y'])

    return train_all_x, train_all_y, test_all_x, test_all_y, user, train_num_sample, test_num_sample, train_file_path, test_file_path

def mutation(dataset, train_all_x, train_all_y, test_all_x, test_all_y, user):

    #os.makedirs('/data/yc/leaf_new_data/femnist/data/train_react_60_translation_15')
    #os.makedirs('/data/yc/leaf_new_data/femnist/data/test_react_60_translation_15')
    
    train_all_x = np.array(train_all_x)
    test_all_x = np.array(test_all_x)

    # change 代表选择多少个client，5%为19, 10%为38， 20%为76， 30%为114， 40%为152， 50%为190 60%-228
    for change in range(228):
        print('=============================')
        print('change clients:' + str(change + 1))
        index_1 = random.randrange(len(test_all_x))

        train_x_value1 = train_all_x[index_1]
        test_x_value1 = test_all_x[index_1]

        train_x_value1 = np.array(train_x_value1)
        test_x_value1 = np.array(test_x_value1)

        # 每次随机决定旋转和平移的大小
        # ro_params = random.choice([10, 20, 30, 40, 50, 60, 70, 80, 90])
        ro_params = random.choice(list(range(-60, 61, 1)))

        # tran_param = random.choice([-1, -2, -3, -4, -5, -6, -7, -8, -9, -10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        # trans_param = random.choice([-1, -2, -3, -4, -5, -6, -7, -8, -9, -10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        tran_param = random.choice(list(range(-15, 16, 1)))
        trans_param = random.choice(list(range(-15, 16, 1)))

        tran_params = [tran_param, trans_param]
        
        #light范围设置小于1变暗 大于1变亮
        light_params = random.choice(list(range(950, 1101, 1))) / 1000

        for img_index_train in range(len(train_x_value1)):
            train_x_value2 = train_x_value1[img_index_train]
            train_x_value2 = np.array(train_x_value2)

            train_x_value2 = train_x_value2.reshape(1, img_rows, img_cols, 1)

            img = train_x_value2 # shape(1,28,28,1)

            # mutate img
            gen_img = deprocess_image(img)# shape(28,28)

            # 大于1变暗和小于1变亮
            if args.operator == 'light':
                # muta_img = exposure.adjust_gamma(gen_img, 5)
                muta_img = exposure.adjust_gamma(gen_img, light_params)
                # imsave('../inputs/demo/init_light_new.png', gen_img)
                # imsave('../inputs/demo/light_change_new.png', muta_img)
            elif args.operator == 'noise':
                muta_img = sp_noise(gen_img, 0.6)
            elif args.operator == 'translation':
                muta_img = image_translation(gen_img, tran_params)
            elif args.operator == 'rotation':
                muta_img = image_rotation(gen_img, ro_params)


            muta_img = muta_img.reshape(-1).tolist()

            train_all_x[index_1][img_index_train] = muta_img

        for img_index_test in range(len(test_x_value1)):

            test_x_value2 = test_x_value1[img_index_test]
            test_x_value2 = np.array(test_x_value2)

            test_x_value2 = test_x_value2.reshape(1, img_rows, img_cols, 1)

            img = test_x_value2

            gen_img = deprocess_image(img)

            if args.operator == 'light':
                muta_img = exposure.adjust_gamma(gen_img, light_params)
            elif args.operator == 'noise':
                muta_img = sp_noise(gen_img, 0.6)
            elif args.operator == 'translation':
                muta_img = image_translation(gen_img, tran_params)
            elif args.operator == 'rotation':
                muta_img = image_rotation(gen_img, ro_params)

            muta_img = muta_img.reshape(-1).tolist()

            test_all_x[index_1][img_index_test] = muta_img

    for i, u in enumerate(user):

        user_data_train[u] = {'x': [], 'y': []}
        user_data_test[u] = {'x': [], 'y': []}

        for j in range(len(train_all_y[i])):
            user_data_train[u]['x'].append(train_all_x[i][j])

            user_data_train[u]['y'].append(train_all_y[i][j])

        for j in range(len(test_all_y[i])):
            user_data_test[u]['x'].append(test_all_x[i][j])

            user_data_test[u]['y'].append(test_all_y[i][j])

    return user_data_train, user_data_test


def wrt_json(user, train_num_sample, test_num_sample, user_data_train, user_data_test, train_file_path,
             test_file_path):
    all_data_train = {}
    all_data_train['users'] = user
    all_data_train['num_samples'] = train_num_sample
    all_data_train['user_data'] = user_data_train

    all_data_test = {}
    all_data_test['users'] = user
    all_data_test['num_samples'] = test_num_sample
    all_data_test['user_data'] = user_data_test

    # print(all_data_test)

    train_file_path = train_file_path.replace('train', 'train_react_60_translation_15')
    test_file_path = test_file_path.replace('test', 'test_react_60_translation_15')

    with open(train_file_path, 'w') as outfile:
        json.dump(all_data_train, outfile)

    with open(test_file_path, 'w') as outfile:
        json.dump(all_data_test, outfile)

train_all_x, train_all_y, test_all_x, test_all_y, user, train_num_sample, test_num_sample, train_file_path, test_file_path = read()
user_data_train, user_data_test = mutation(args.dataset, train_all_x, train_all_y, test_all_x, test_all_y, user)
wrt_json(user, train_num_sample, test_num_sample, user_data_train, user_data_test, train_file_path, test_file_path)