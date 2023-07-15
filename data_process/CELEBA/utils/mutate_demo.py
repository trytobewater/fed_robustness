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


def mutation():
    tran_param = random.choice(list(range(-60, 61, 1)))
    trans_param = random.choice(list(range(-60, 61, 1)))

    tran_params = [tran_param, trans_param]
    gen_img_dir = os.path.join('/data', 'yc', 'leaf_new_data', 'celeba', 'data', 't_demo',
                                   '000001.jpg')
    gen_img = cv2.imread(gen_img_dir)
    muta_img = image_translation(gen_img, tran_params)
    muta_img_dir = os.path.join('/data', 'yc', 'leaf_new_data', 'celeba', 'data','t_demo',
                                   '000001.jpg')

    cv2.imwrite(muta_img_dir, muta_img)

    return 0






mutation()
