import random
from collections import defaultdict

import numpy as np
from keras import backend as K
from keras.models import Model
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input

import os
from PIL import Image
import cv2

# util function to convert a tensor into a valid image
# light 明暗  occl 一个矩形 blackout 黑色小块
def deprocess_image(x):
    # x = x.reshape((84, 84, 3))
    # # Remove zero-center by mean pixel
    # x[:, :, 0] += 103.939
    # x[:, :, 1] += 116.779
    # x[:, :, 2] += 123.68
    x = np.clip(x, 0, 255).astype('uint8')
    return x.reshape(x.shape[1], x.shape[2], x.shape[3])

def preprocess_image(img_path):#218 178
    img = Image.open(os.path.join(img_path))
    img = img.resize((84, 84)).convert('RGB')
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

    return img

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)


def constraint_occl(gradients, start_point, rect_shape):
    new_grads = np.zeros_like(gradients)
    new_grads[:, start_point[0]:start_point[0] + rect_shape[0],
    start_point[1]:start_point[1] + rect_shape[1]] = gradients[:, start_point[0]:start_point[0] + rect_shape[0],
                                                     start_point[1]:start_point[1] + rect_shape[1]]
    return new_grads


def constraint_light(gradients):
    new_grads = np.ones_like(gradients)
    grad_mean = 1e4 * np.mean(gradients)
    return grad_mean * new_grads


def constraint_mask(gen_img, x, y, w, h, size):
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

def constraint_dark(gradients):
  new_grads = np.zeros_like(gradients)
  grad_mean = np.mean(gradients)
  return grad_mean * new_grads

# 加污渍
def constraint_black(gradients, rect_shape=(15, 15)):
  start_point = (
      random.randint(0, gradients.shape[1] - rect_shape[0]), random.randint(0, gradients.shape[2] - rect_shape[1]))
  new_grads = np.zeros_like(gradients)
  patch = gradients[:, start_point[0]:start_point[0] + rect_shape[0], start_point[1]:start_point[1] + rect_shape[1]]
  if np.mean(patch) < 0:
      new_grads[:, start_point[0]:start_point[0] + rect_shape[0],
      start_point[1]:start_point[1] + rect_shape[1]] = -np.ones_like(patch)
  return new_grads


def init_coverage_tables(model1):
  model_layer_dict1 = defaultdict(bool)
  # model_layer_dict2 = defaultdict(bool)
  # model_layer_dict3 = defaultdict(bool)
  init_dict(model1, model_layer_dict1)
  # init_dict(model2, model_layer_dict2)
  # init_dict(model3, model_layer_dict3)
  return model_layer_dict1


def init_dict(model, model_layer_dict):
  for layer in model.layers:
      if 'flatten' in layer.name or 'input' in layer.name:
          continue
      for index in range(layer.output_shape[-1]):
          model_layer_dict[(layer.name, index)] = False


def neuron_to_cover(model_layer_dict):
  not_covered = [(layer_name, index) for (layer_name, index), v in model_layer_dict.items() if not v]
  if not_covered:
      layer_name, index = random.choice(not_covered)
  else:
      layer_name, index = random.choice(model_layer_dict.keys())
  return layer_name, index


def neuron_covered(model_layer_dict):
  covered_neurons = len([v for v in model_layer_dict.values() if v])
  total_neurons = len(model_layer_dict)
  return covered_neurons, total_neurons, covered_neurons / float(total_neurons)


def update_coverage(input_data, model, model_layer_dict, threshold=0):
  layer_names = [layer.name for layer in model.layers if
                 'flatten' not in layer.name and 'input' not in layer.name]

  intermediate_layer_model = Model(inputs=model.input,
                                   outputs=[model.get_layer(layer_name).output for layer_name in layer_names])
  intermediate_layer_outputs = intermediate_layer_model.predict(input_data)

  for i, intermediate_layer_output in enumerate(intermediate_layer_outputs):
      scaled = scale(intermediate_layer_output[0])
      for num_neuron in range(scaled.shape[-1]):
          if np.mean(scaled[..., num_neuron]) > threshold and not model_layer_dict[(layer_names[i], num_neuron)]:
              model_layer_dict[(layer_names[i], num_neuron)] = True


def full_coverage(model_layer_dict):
  if False in model_layer_dict.values():
      return False
  return True


def scale(intermediate_layer_output, rmax=1, rmin=0):
  X_std = (intermediate_layer_output - intermediate_layer_output.min()) / (
      intermediate_layer_output.max() - intermediate_layer_output.min())
  X_scaled = X_std * (rmax - rmin) + rmin
  return X_scaled


def fired(model, layer_name, index, input_data, threshold=0):
  intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
  intermediate_layer_output = intermediate_layer_model.predict(input_data)[0]
  scaled = scale(intermediate_layer_output)
  if np.mean(scaled[..., index]) > threshold:
      return True
  return False


def diverged(predictions1, predictions2, predictions3, target):
  #     if predictions2 == predictions3 == target and predictions1 != target:
  if not predictions1 == predictions2 == predictions3:
      return True
  return False
