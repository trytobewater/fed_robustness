import numpy as np
# import tensorflow as tf
from keras import backend as K
from keras.layers import Input
import argparse
from model_utils import read_data
from operate_utils import *
# import operate_utils
import random
from genarate_model1 import Model1
import json
import cv2

import os

DATASETS = ['sent140', 'femnist', 'shakespeare', 'celeba', 'synthetic', 'reddit']

parser = argparse.ArgumentParser(description='Main function for difference-inducing input generation in FEMNIST dataset')
# parser.add_argument('grad_iterations', help="number of iterations of gradient descent", type=int, default=20)
parser.add_argument('-dataset',
                    help='name of dataset;',
                    type=str,
                    choices=DATASETS,
                    default='celeba')

parser.add_argument('-operate_1', help="realistic transformation type", choices=['translation', 'rotation'], default='translation')
parser.add_argument('-operate_2', help="realistic transformation type", choices=['translation', 'rotation'], default='rotation')
parser.add_argument('-operate_3', help="realistic transformation type", choices=['light', 'occl', 'blackout', 'whiteout'], default='blackout')
parser.add_argument('-operate_4', help="realistic transformation type", choices=['light', 'occl', 'blackout', 'whiteout'], default='whiteout')

args = parser.parse_args()

img_rows = 84
img_cols = 84

input_shape = (img_rows, img_cols, 3)
input_tensor = Input(shape=input_shape)

# load multiple models sharing same input tensor
model = Model1(input_tensor=input_tensor)
model_layer_dict1 = init_coverage_tables(model)

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

def mutation(dataset, train_all_x, train_all_y, test_all_x, test_all_y, user):

    train_all_x = np.array(train_all_x)
    test_all_x = np.array(test_all_x)

    # print(len(test_all_x))
    # print(len(train_all_x))

    # change 代表选择多少个client，5%为19, 10%为38， 20%为76
    for change in range(42):

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

            gen_img_dir = os.path.join(root_pa, 'yc', 'leaf_te_data', dataset, 'data', 'raw', 'img_align_celeba_translation_5', train_x_value2)

            # gen_img_deprocessed.shape(218,178,3)
            gen_img_deprocessed = cv2.imread(gen_img_dir)
            # cv2.imwrite('../data/inputs/rotation_init.png', gen_img_deprocessed)

            # imsave('../inputs/translation_init.png', gen_img_deprocessed)

            if args.operate_1 == 'rotation':
                # params为旋转角度
                # ro_params = 60
                muta_img_deprocessed = image_rotation(gen_img_deprocessed, ro_params)
            elif args.operate_1 == 'translation':
                # params为平移距离
                # tran_params = [10, 10]
                muta_img_deprocessed = image_translation(gen_img_deprocessed, tran_params)

            # imsave('../inputs/translation_' + str(tran_param) + '.png', muta_img_deprocessed)
            # cv2.imwrite('../data/inputs/rotation_init' + str(tran_param) + '.png', muta_img_deprocessed)
            #local data
            # muta_img_dir = os.path.join('..', 'data', 'raw', 'img_align_celeba_translation_5', train_x_value2)

            #gpu data
            muta_img_dir = os.path.join(root_pa, 'yc', 'leaf_te_data', dataset, 'data', 'raw', 'img_align_celeba_translation_5', train_x_value2)

            #在突变图片目录把突变后的图片覆盖原图片
            cv2.imwrite(muta_img_dir, muta_img_deprocessed)

        for img_index_1 in range(len(test_x_value1)):

            test_x_value2 = test_x_value1[img_index_1]

            gen_img_dir = os.path.join(root_pa, 'yc', 'leaf_te_data', dataset, 'data', 'raw', 'img_align_celeba', test_x_value2)

            gen_img_deprocessed = cv2.imread(gen_img_dir)

            if args.operate_1 == 'rotation':
                # params为旋转角度
                # ro_params = 60
                muta_img_deprocessed = image_rotation(gen_img_deprocessed, ro_params)
            elif args.operate_1 == 'translation':
                # params为平移距离
                # tran_params = [10, 10]
                muta_img_deprocessed = image_translation(gen_img_deprocessed, tran_params)

            #将转好的图片重新归一化为0,1tensor
                muta_img_dir = os.path.join(root_pa, 'yc', 'leaf_te_data', dataset, 'data', 'raw', 'img_align_celeba_translation_5', test_x_value2)

            # 在突变图片目录把突变后的图片覆盖原图片
            cv2.imwrite(muta_img_dir, muta_img_deprocessed)

    for change in range(42):

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

            gen_img_dir = os.path.join(root_pa, 'yc', 'leaf_te_data', dataset, 'data', 'raw', 'img_align_celeba_translation_5', train_x_value2)

            # gen_img_deprocessed.shape(218,178,3)
            gen_img_deprocessed = cv2.imread(gen_img_dir)
            # cv2.imwrite('../data/inputs/rotation_init.png', gen_img_deprocessed)

            # imsave('../inputs/translation_init.png', gen_img_deprocessed)

            if args.operate_2 == 'rotation':
                # params为旋转角度
                # ro_params = 60
                muta_img_deprocessed = image_rotation(gen_img_deprocessed, ro_params)
            elif args.operate_2 == 'translation':
                # params为平移距离
                # tran_params = [10, 10]
                muta_img_deprocessed = image_translation(gen_img_deprocessed, tran_params)

            # imsave('../inputs/translation_' + str(tran_param) + '.png', muta_img_deprocessed)
            # cv2.imwrite('../data/inputs/rotation_init' + str(tran_param) + '.png', muta_img_deprocessed)
            #local data
            # muta_img_dir = os.path.join('..', 'data', 'raw', 'img_align_celeba_translation_5', train_x_value2)

            #gpu data
            muta_img_dir = os.path.join(root_pa, 'yc', 'leaf_te_data', dataset, 'data', 'raw', 'img_align_celeba_translation_5', train_x_value2)

            #在突变图片目录把突变后的图片覆盖原图片
            cv2.imwrite(muta_img_dir, muta_img_deprocessed)

        for img_index_1 in range(len(test_x_value1)):

            test_x_value2 = test_x_value1[img_index_1]

            gen_img_dir = os.path.join(root_pa, 'yc', 'leaf_te_data', dataset, 'data', 'raw', 'img_align_celeba', test_x_value2)

            gen_img_deprocessed = cv2.imread(gen_img_dir)

            if args.operate_2 == 'rotation':
                # params为旋转角度
                # ro_params = 60
                muta_img_deprocessed = image_rotation(gen_img_deprocessed, ro_params)
            elif args.operate_2 == 'translation':
                # params为平移距离
                # tran_params = [10, 10]
                muta_img_deprocessed = image_translation(gen_img_deprocessed, tran_params)

            #将转好的图片重新归一化为0,1tensor
                muta_img_dir = os.path.join(root_pa, 'yc', 'leaf_te_data', dataset, 'data', 'raw', 'img_align_celeba_translation_5', test_x_value2)

            # 在突变图片目录把突变后的图片覆盖原图片
            cv2.imwrite(muta_img_dir, muta_img_deprocessed)

    for change in range(42):

        index_1 = random.randrange(len(test_all_x))

        train_x_value1 = train_all_x[index_1]
        test_x_value1 = test_all_x[index_1]

        train_x_value1 = np.array(train_x_value1)
        test_x_value1 = np.array(test_x_value1)
        # print(len(test_all_x))

        root_pa = '/data'

        for img_index in range(len(train_x_value1)):

            train_x_value2 = train_x_value1[img_index]

            # local data
            # gen_img_dir = os.path.join('..', 'data', 'raw', 'img_align_celeba', train_x_value2)

            gen_img_dir = os.path.join(root_pa, 'yc', 'leaf_te_data', dataset, 'data', 'raw', 'img_align_celeba',
                                       train_x_value2)

            gen_img_init = preprocess_image(gen_img_dir)
            # cv2.imwrite('../data/inputs/light_init.png', gen_img)

            gen_img = np.array(gen_img_init)
            gen_img = gen_img.astype('float64')

            gen_img = np.expand_dims(gen_img, axis=0)
            orig_img = gen_img.copy()

            orig_label = np.argmax(model.predict(gen_img)[0])
            layer_name1, index1 = neuron_to_cover(model_layer_dict1)

            loss = -0.1 * K.mean(model.get_layer('before_softmax').output[
                                     ..., orig_label])  # -args.weight_diff * K.mean(model.get_layer('before_softmax').output[..., orig_label])

            loss1_neuron = K.mean(model.get_layer(layer_name1).output[..., index1])

            layer_output = loss + 0.1 * (loss1_neuron)  # loss + args.weight_nc * (loss1_neuron)

            # for adversarial image generation
            final_loss = K.mean(layer_output)

            # we compute the gradient of the input picture wrt this loss
            grads = normalize(K.gradients(final_loss, input_tensor)[0])

            # this function returns the loss and grads given the input picture
            iterate = K.function([input_tensor], [loss, loss1_neuron, grads])

            if args.transformation == 'light':
                for iters in range(40):
                    loss_value1, loss_neuron1, grads_value = iterate([gen_img])
                    grads_value = constraint_light(grads_value)
                    gen_img += grads_value * 0.2
                gen_img_deprocessed = deprocess_image(gen_img)

            elif args.transformation == 'mask':
                gen_img_deprocessed = constraint_mask(gen_img_init, 10, 10, 84 - 35, 84 - 35, 4)
            # cv2.imwrite('../data/inputs/light_' + str(start) + '_' + str(occl) + '_.png', gen_img_deprocessed)
            # cv2.imwrite('../data/inputs/light_muteta.png', gen_img_deprocessed)

            muta_img_dir = os.path.join(root_pa, 'yc', 'leaf_te_data', dataset, 'data', 'raw',
                                        'img_align_celeba_light_5', train_x_value2)

            # 在突变图片目录把突变后的图片覆盖原图片
            cv2.imwrite(muta_img_dir, gen_img_deprocessed)

        for img_index_1 in range(len(test_x_value1)):

            test_x_value2 = test_x_value1[img_index_1]

            # local data
            # muta_img_dir = os.path.join('..', 'data', 'raw', 'img_align_celeba', train_x_value2)

            gen_img_dir = os.path.join(root_pa, 'yc', 'leaf_te_data', dataset, 'data', 'raw', 'img_align_celeba',
                                       test_x_value2)

            gen_img_init = preprocess_image(gen_img_dir)
            # cv2.imwrite('../data/inputs/light_init.png', gen_img)

            gen_img = np.array(gen_img_init)
            gen_img = gen_img.astype('float64')
            gen_img = np.expand_dims(gen_img, axis=0)
            orig_img = gen_img.copy()

            # imsave('../inputs/translation_init.png', gen_img_deprocessed)
            orig_label = np.argmax(model.predict(gen_img)[0])
            layer_name1, index1 = neuron_to_cover(model_layer_dict1)

            loss = -0.1 * K.mean(model.get_layer('before_softmax').output[
                                     ..., orig_label])  # -args.weight_diff * K.mean(model.get_layer('before_softmax').output[..., orig_label])

            loss1_neuron = K.mean(model.get_layer(layer_name1).output[..., index1])

            layer_output = loss + 0.1 * (loss1_neuron)  # loss + args.weight_nc * (loss1_neuron)

            # for adversarial image generation
            final_loss = K.mean(layer_output)

            # we compute the gradient of the input picture wrt this loss
            grads = normalize(K.gradients(final_loss, input_tensor)[0])

            # this function returns the loss and grads given the input picture
            iterate = K.function([input_tensor], [loss, loss1_neuron, grads])

            if args.transformation == 'light':
                for iters in range(40):
                    loss_value1, loss_neuron1, grads_value = iterate([gen_img])
                    grads_value = constraint_light(grads_value)
                    gen_img += grads_value * 0.2
                gen_img_deprocessed = deprocess_image(gen_img)

            elif args.transformation == 'mask':
                gen_img_deprocessed = constraint_mask(gen_img_init, 10, 10, 84 - 35, 84 - 35, 4)
            # imsave('../inputs/light_' + str(start) + '_' + str(occl) + '_.png', gen_img_deprocessed)

            muta_img_dir = os.path.join(root_pa, 'yc', 'leaf_te_data', dataset, 'data', 'raw',
                                        'img_align_celeba_light_5', test_x_value2)

            # 在突变图片目录把突变后的图片覆盖原图片
            cv2.imwrite(muta_img_dir, gen_img_deprocessed)

    for change in range(42):

        index_1 = random.randrange(len(test_all_x))

        train_x_value1 = train_all_x[index_1]
        test_x_value1 = test_all_x[index_1]

        train_x_value1 = np.array(train_x_value1)
        test_x_value1 = np.array(test_x_value1)
        # print(len(test_all_x))

        start = random.choice([6, 7, 8])
        # occlusion size
        occl = random.choice([6, 7, 8])
        start_point = (start, start)
        occl_size = (occl, occl)

        root_pa = '/data'

        for img_index in range(len(train_x_value1)):

            train_x_value2 = train_x_value1[img_index]

            # local data
            # gen_img_dir = os.path.join('..', 'data', 'raw', 'img_align_celeba', train_x_value2)

            gen_img_dir = os.path.join(root_pa, 'yc', 'leaf_te_data', dataset, 'data', 'raw', 'img_align_celeba',
                                       train_x_value2)

            gen_img_init = preprocess_image(gen_img_dir)
            # cv2.imwrite('../data/inputs/light_init.png', gen_img)

            gen_img = np.array(gen_img_init)
            gen_img = gen_img.astype('float64')

            gen_img = np.expand_dims(gen_img, axis=0)
            orig_img = gen_img.copy()

            orig_label = np.argmax(model.predict(gen_img)[0])
            layer_name1, index1 = neuron_to_cover(model_layer_dict1)

            loss = -0.1 * K.mean(model.get_layer('before_softmax').output[
                                     ..., orig_label])  # -args.weight_diff * K.mean(model.get_layer('before_softmax').output[..., orig_label])

            loss1_neuron = K.mean(model.get_layer(layer_name1).output[..., index1])

            layer_output = loss + 0.1 * (loss1_neuron)  # loss + args.weight_nc * (loss1_neuron)

            # for adversarial image generation
            final_loss = K.mean(layer_output)

            # we compute the gradient of the input picture wrt this loss
            grads = normalize(K.gradients(final_loss, input_tensor)[0])

            # this function returns the loss and grads given the input picture
            iterate = K.function([input_tensor], [loss, loss1_neuron, grads])

            if args.transformation == 'light':
                for iters in range(40):
                    loss_value1, loss_neuron1, grads_value = iterate([gen_img])
                    grads_value = constraint_light(grads_value)
                    gen_img += grads_value * 0.2
                gen_img_deprocessed = deprocess_image(gen_img)

            elif args.transformation == 'mask':
                gen_img_deprocessed = constraint_mask(gen_img_init, 10, 10, 84 - 35, 84 - 35, 4)
            # cv2.imwrite('../data/inputs/light_' + str(start) + '_' + str(occl) + '_.png', gen_img_deprocessed)
            # cv2.imwrite('../data/inputs/light_muteta.png', gen_img_deprocessed)

            muta_img_dir = os.path.join(root_pa, 'yc', 'leaf_te_data', dataset, 'data', 'raw',
                                        'img_align_celeba_light_5', train_x_value2)

            # 在突变图片目录把突变后的图片覆盖原图片
            cv2.imwrite(muta_img_dir, gen_img_deprocessed)

        for img_index_1 in range(len(test_x_value1)):

            test_x_value2 = test_x_value1[img_index_1]

            # local data
            # muta_img_dir = os.path.join('..', 'data', 'raw', 'img_align_celeba', train_x_value2)

            gen_img_dir = os.path.join(root_pa, 'yc', 'leaf_te_data', dataset, 'data', 'raw', 'img_align_celeba',
                                       test_x_value2)

            gen_img_init = preprocess_image(gen_img_dir)
            # cv2.imwrite('../data/inputs/light_init.png', gen_img)

            gen_img = np.array(gen_img_init)
            gen_img = gen_img.astype('float64')
            gen_img = np.expand_dims(gen_img, axis=0)
            orig_img = gen_img.copy()

            # imsave('../inputs/translation_init.png', gen_img_deprocessed)
            orig_label = np.argmax(model.predict(gen_img)[0])
            layer_name1, index1 = neuron_to_cover(model_layer_dict1)

            loss = -0.1 * K.mean(model.get_layer('before_softmax').output[
                                     ..., orig_label])  # -args.weight_diff * K.mean(model.get_layer('before_softmax').output[..., orig_label])

            loss1_neuron = K.mean(model.get_layer(layer_name1).output[..., index1])

            layer_output = loss + 0.1 * (loss1_neuron)  # loss + args.weight_nc * (loss1_neuron)

            # for adversarial image generation
            final_loss = K.mean(layer_output)

            # we compute the gradient of the input picture wrt this loss
            grads = normalize(K.gradients(final_loss, input_tensor)[0])

            # this function returns the loss and grads given the input picture
            iterate = K.function([input_tensor], [loss, loss1_neuron, grads])

            if args.transformation == 'light':
                for iters in range(40):
                    loss_value1, loss_neuron1, grads_value = iterate([gen_img])
                    grads_value = constraint_light(grads_value)
                    gen_img += grads_value * 0.2
                gen_img_deprocessed = deprocess_image(gen_img)

            elif args.transformation == 'mask':
                gen_img_deprocessed = constraint_mask(gen_img_init, 10, 10, 84 - 35, 84 - 35, 4)

            # imsave('../inputs/light_' + str(start) + '_' + str(occl) + '_.png', gen_img_deprocessed)

            muta_img_dir = os.path.join(root_pa, 'yc', 'leaf_te_data', dataset, 'data', 'raw',
                                        'img_align_celeba_light_5', test_x_value2)

            # 在突变图片目录把突变后的图片覆盖原图片
            cv2.imwrite(muta_img_dir, gen_img_deprocessed)


train_all_x, test_all_x = read()
mutation(args.dataset, train_all_x, test_all_x)
