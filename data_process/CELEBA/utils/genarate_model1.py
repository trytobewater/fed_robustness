'''
LeNet-5
'''

# usage: python MNISTModel1.py - train the model

from __future__ import print_function
import tensorflow as tf
# from keras.datasets import mnist
from keras.layers import Convolution2D, MaxPooling2D, Input, Dense, Activation, Flatten
from keras.models import Model
from keras.utils import to_categorical
import numpy as np
# from configs import bcolors
from model_utils import read_data
import os
from PIL import Image
import cv2

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

IMAGES_DIR = os.path.join('/data', 'yc', 'leaf_te_data', 'celeba', 'data', 'raw', 'img_align_celeba')

#local data
# IMAGES_DIR = os.path.join( '../data', 'raw', 'img_align_celeba')
IMAGE_SIZE = 84

def Model1(input_tensor=None, train=False):
    nb_classes = 2
    # convolution kernel size
    kernel_size = (5, 5)

    if train:
        batch_size = 256
        nb_epoch = 25

        # the data, shuffled and split between train and test sets
        # (x_train, y_train), (x_test, y_test) = mnist.load_data()

        # root_pa = '/data'
        # dataset = 'celeba'
        # train_data_dir = os.path.join(root_pa, 'yc', 'leaf_te_data', dataset, 'data', 'train')
        # test_data_dir = os.path.join(root_pa, 'yc', 'leaf_te_data', dataset, 'data', 'test')

        train_data_dir = '../data/train'
        test_data_dir = '../data/test'

        users, groups, train_data, test_data = read_data(train_data_dir, test_data_dir, 0)
        train_da = []
        test_da = []

        for u in users:
            # print(u)
            train_da.append(train_data[u])
            test_da.append(test_data[u])

        x_train, y_train, x_test, y_test = femnist_loaders(train_da, test_da)

        # print(len(x_train))

        # print(x_train.shape)
        x_train = process_x(x_train)
        x_test = process_x(x_test)
        # print(x_test.shape[0])
        input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)

        # x_train = x_train.astype('float32')
        # x_test = x_test.astype('float32')
        # x_train /= 255
        # x_test /= 255

        #delete
        # convert class vectors to binary class matrices
        y_train = to_categorical(y_train, nb_classes)
        y_test = to_categorical(y_test, nb_classes)

        input_tensor = Input(shape=input_shape)
    #delete
    # elif input_tensor is None:
    #     print(bcolors.FAIL + 'you have to proved input_tensor when testing')
    #     exit()

    x = Convolution2D(24, (5, 5), padding='valid', activation='relu', strides=(2, 2), name='block1_conv1')(input_tensor)
    x = Convolution2D(36, (5, 5), padding='valid', activation='relu', strides=(2, 2), name='block1_conv2')(x)
    x = Convolution2D(48, (5, 5), padding='valid', activation='relu', strides=(2, 2), name='block1_conv3')(x)
    x = Convolution2D(64, (3, 3), padding='valid', activation='relu', strides=(1, 1), name='block1_conv4')(x)
    x = Convolution2D(64, (3, 3), padding='valid', activation='relu', strides=(1, 1), name='block1_conv5')(x)
    x = Flatten(name='flatten')(x)
    x = Dense(1164, activation='relu', name='fc1')(x)
    x = Dense(100, activation='relu', name='fc2')(x)
    x = Dense(50, activation='relu', name='fc3')(x)
    x = Dense(10, activation='relu', name='fc4')(x)
    x = Dense(nb_classes, name='before_softmax')(x)
    x = Activation('softmax', name='predictions')(x)

    model = Model(input_tensor, x)

    if train:
        # compiling
        model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

        # trainig
        model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=batch_size, epochs=nb_epoch, verbose=1)
        # save model
        model.save_weights('./Model1.h5')
        score = model.evaluate(x_test, y_test, verbose=0)
        print('\n')
        print('Overall Test score:', score[0])
        print('Overall Test accuracy:', score[1])
    else:
        model.load_weights('./Model1.h5')
        # print(bcolors.OKBLUE + 'Model1 loaded' + bcolors.ENDC)

    return model

def femnist_loaders(train_data={'x' : [],'y' : []}, test_data={'x' : [],'y' : []}):
    train_x = []
    train_y = []
    test_x = []
    test_y = []

    for data in train_data:
        for x in data['x']:
            train_x.append(x)
        for y in data['y']:
            train_y.append(y)
    for data in test_data:
        for x in data['x']:
            test_x.append(x)
        for y in data['y']:
            test_y.append(y)

    train_x = np.array(train_x)
    train_y = np.array(train_y)
    test_x = np.array(test_x)
    test_y = np.array(test_y)

    return train_x, train_y, test_x, test_y

def process_x(raw_x_batch):
    x_batch = [_load_image(i) for i in raw_x_batch]
    x_batch = np.array(x_batch)
    return x_batch

def process_y(raw_y_batch):
    return raw_y_batch

def _load_image(img_name):
    img = Image.open(os.path.join(IMAGES_DIR, img_name))
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE)).convert('RGB')
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    return np.array(img)

if __name__ == '__main__':
    Model1(train=True)
