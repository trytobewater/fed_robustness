import json
import os
import shutil
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

def copy_file(source,target):
    source_path = os.path.abspath(source)
    target_path = os.path.abspath(target)

    if not os.path.exists(target_path):
        os.makedirs(target_path)

    if os.path.exists(source_path):
        # root 所指的是当前正在遍历的这个文件夹的本身的地址
        # dirs 是一个 list，内容是该文件夹中所有的目录的名字(不包括子目录)
        # files 同样是 list, 内容是该文件夹中所有的文件(不包括子目录)
        for root, dirs, files in os.walk(source_path):
            for file in files:
                src_file = os.path.join(root, file)
                shutil.copy(src_file, target_path)


    print('copy files finished!')


def mutation(dataset, train_all_x, test_all_x):
    t_dir='train_react_60_light_0.1-10'
    te_dir='test_react_60_light_0.1-10'

    copy_file('/data/yc/leaf_new_data/celeba/data/train_need_data','/data/yc/leaf_new_data/celeba/data/'+t_dir)
    copy_file('/data/yc/leaf_new_data/celeba/data/test_need_data', '/data/yc/leaf_new_data/celeba/data/' + te_dir)


    train_all_x = np.array(train_all_x)
    test_all_x = np.array(test_all_x)

    # change 代表选择多少个client，10%为94，  30%为282， 60%为563,统一向上取整 共937个
    for change in range(563):
        print('=============================')
        print('change clients:' + str(change + 1))
        index_1 = random.randrange(len(test_all_x))

        train_x_value1 = train_all_x[index_1]
        test_x_value1 = test_all_x[index_1]

        train_x_value1 = np.array(train_x_value1)
        test_x_value1 = np.array(test_x_value1)

        #light范围设置小于1变暗 大于1变亮
        light_params = random.choice(list(range(100, 10001, 1))) / 1000

        # 每次随机决定旋转和平移的大小
        # ro_params = random.choice([10, 20, 30, 40, 50, 60, 70, 80, 90])
        ro_params = random.choice(list(range(-60, 61, 1)))

        # tran_param = random.choice([-1, -2, -3, -4, -5, -6, -7, -8, -9, -10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        # trans_param = random.choice([-1, -2, -3, -4, -5, -6, -7, -8, -9, -10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        tran_param = random.choice(list(range(-60, 61, 1)))
        trans_param = random.choice(list(range(-60,61, 1)))

        tran_params = [tran_param, trans_param]

        noi_params=0.2

        mask_x = random.choice(list(range(60, 91, 1)))
        mask_y = random.choice(list(range(80, 121, 1)))
        mask_h = random.choice(list(range(20, 81, 1)))
        mask_w = random.choice(list(range(20, 81, 1)))
        mask_size = random.choice(list(range(3, 7, 1)))

        for img_index_train in range(len(train_x_value1)):

            train_x_value2 = train_x_value1[img_index_train]

            # local
            gen_img_dir = os.path.join( '/data','yc','leaf_new_data','celeba','data', 'train_need_data', train_x_value2)

            # 501
            # gen_img_dir = os.path.join('/data2', 'yc', 'leaf_te_data', dataset, 'data', 'raw',
            #                            'img_align_celeba_noise_40', train_x_value2)

            # 519
            # gen_img_dir = os.path.join('/data6T', 'yangchen', 'leaf-tes', 'data', dataset, 'data', 'raw',
            #                                 'img_align_celeba_mask_40', train_x_value2)

            gen_img = cv2.imread(gen_img_dir)
            # cv2.imwrite('../inputs/new_demo/mask_init.png', gen_img)
            # 大于1变暗和小于1变亮
            if args.operator == 'light':
                # muta_img = exposure.adjust_gamma(gen_img, 5)
                muta_img = exposure.adjust_gamma(gen_img, light_params)
            elif args.operator == 'noise':
                muta_img = sp_noise(gen_img, noi_params)
            elif args.operator == 'translation':
                muta_img = image_translation(gen_img, tran_params)
            elif args.operator == 'rotation':
                muta_img = image_rotation(gen_img, ro_params)
            elif args.operator == 'mask':
                muta_img = image_mask(gen_img, mask_x, mask_y, mask_h, mask_w, mask_size)

            # cv2.imwrite('../inputs/new_demo/mask_change.png', muta_img)

            # # 501
            # muta_img_dir = os.path.join('/data2', 'yc', 'leaf_te_data', dataset, 'data', 'raw',
            #                            'img_align_celeba_noise_40', train_x_value2)

            # 519
            muta_img_dir = os.path.join('/data','yc','leaf_new_data','celeba','data',  t_dir, train_x_value2)

            cv2.imwrite(muta_img_dir, muta_img)

        for img_index_test in range(len(test_x_value1)):

            test_x_value2 = test_x_value1[img_index_test]


            # # 501
            # gen_img_dir = os.path.join('/data2', 'yc', 'leaf_te_data', dataset, 'data', 'raw',
            #                            'img_align_celeba_noise_40', test_x_value2)

            # 519
            gen_img_dir = os.path.join('/data','yc','leaf_new_data','celeba','data', 'test_need_data', test_x_value2)

            gen_img = cv2.imread(gen_img_dir)

            if args.operator == 'light':
                muta_img = exposure.adjust_gamma(gen_img, light_params)
            elif args.operator == 'noise':
                muta_img = sp_noise(gen_img, noi_params)
            elif args.operator == 'translation':
                muta_img = image_translation(gen_img, tran_params)
            elif args.operator == 'rotation':
                muta_img = image_rotation(gen_img, ro_params)
            elif args.operator == 'mask':
                muta_img = image_mask(gen_img, mask_x, mask_y, mask_h, mask_w, mask_size)

            # 501
            # muta_img_dir = os.path.join('/data2', 'yc', 'leaf_te_data', dataset, 'data', 'raw',
            #                             'img_align_celeba_noise_40', test_x_value2)
            # 519
            muta_img_dir = os.path.join('/data','yc','leaf_new_data','celeba','data', te_dir, test_x_value2)

            cv2.imwrite(muta_img_dir, muta_img)


train_all_x, test_all_x = read()
mutation(args.dataset, train_all_x, test_all_x)
