# A. 平移, 往上下左右四个方向平移图像, 但是平移度不能超过10\%, 平移之后的部分通过某种方式default地填充 (这个需要自己实现, 建议参考F 操作进行实现)
# B. 中心部分裁剪(CenterCrop) #
# C. 旋转一定的度数, 但是要小于20度 (RandomRotation)
# -D. 调整大小（Resizing）
# E. 调整对比度 (RandomContrast)
# F. 裁剪图像 (RandomCrop)
# G. 水平和垂直翻转 (RandomFlip)
# H. 缩放图像 (RandomZoom )
# -I. 调整高度 (RandomHeight)  剪裁图片吗??
# -J. 调整宽度 (RandomWidth)
# K. 变化数据中若干位bit的值, 即对数据中心随机选取若干bit, 进行flip (bitflip)
# L. 变化数据中某个 (或者某几个) 属性的值 (AttChange)
import os
import random

import tensorflow as tf
from keras.datasets import mnist, fashion_mnist, cifar10
from keras_preprocessing.image import ImageDataGenerator

import SVNH_DatasetUtil

tf.enable_eager_execution()
import numpy as np
from tqdm import tqdm


# 扩增类
class daugor(object):
    def __init__(self, params, ):
        self.params = params
        self.data_name = params["data_name"]
        if self.data_name == "mnist":
            (x_train, y_train), (x_test, y_test) = self.load_mnist()
        elif self.data_name == "fashion":
            (x_train, y_train), (x_test, y_test) = self.load_fashion()
        elif self.data_name == "cifar":
            (x_train, y_train), (x_test, y_test) = self.load_cifar()
        else:
            (x_train, y_train), (x_test, y_test) = self.load_svhn()
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.x_path = self.params["base_dir"] + "/" + 'x_{}'
        self.y_path = self.params["base_dir"] + "/" + 'y_{}'
        self.op_path = self.params["base_dir"] + "/" + 'op_{}'
        self.gen = None
        self.init_dir()
        self.init_gen()
        # 初始化文件夹

    def init_dir(self):
        if not os.path.exists(self.params["base_dir"]):
            os.makedirs(self.params["base_dir"])

    def init_gen(self):
        params_map = {
            "w_r": 0.1,  # 左右平移
            "h_r": 0.1,  # 上下平移
            "rotation_range": 20,  # 旋转角度
            "zoom_range": 0.2,  # 随机缩放
            "brightness_range": [0.5, 1.5]
        }

        gen = ImageDataGenerator(width_shift_range=params_map['w_r'], height_shift_range=params_map['h_r'],
                                 rotation_range=params_map['rotation_range'], zoom_range=params_map['zoom_range'],
                                 fill_mode="constant", brightness_range=params_map["brightness_range"])
        self.gen = gen

    # 获得操作名称
    def get_op_name(self, op):
        # A 平移 B 中心裁剪 C 旋转 D调整亮度 E 调整对比度 F随机裁剪  G翻转 H 随机缩放
        op_map = {
            "A": '平移',
            "B": '中心裁剪',
            "C": '旋转',
            "D": '调整亮度',
            "E": '调整对比度',
            "F": '随机裁剪',
            "G": '翻转',
            "H": '随机缩放',
        }
        return op_map[op]

    # 加载数据集
    def load_mnist(self):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_test = x_test.astype('float32').reshape(-1, 28, 28, 1)
        x_train = x_train.astype('float32').reshape(-1, 28, 28, 1)
        return (x_train, y_train), (x_test, y_test)

    def load_fashion(self):
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        x_test = x_test.astype('float32').reshape(-1, 28, 28, 1)
        x_train = x_train.astype('float32').reshape(-1, 28, 28, 1)
        return (x_train, y_train), (x_test, y_test)

    def load_cifar(self):
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_test = x_test.astype('float32').reshape(-1, 32, 32, 3)
        x_train = x_train.astype('float32').reshape(-1, 32, 32, 3)
        return (x_train, y_train), (x_test, y_test)

    def load_svhn(self):
        (X_train, Y_train), (X_test, Y_test) = SVNH_DatasetUtil.load_data()  # 32*32
        Y_test = np.argmax(Y_test, axis=1)
        Y_train = np.argmax(Y_train, axis=1)
        return (X_train, Y_train), (X_test, Y_test)

    # 扩增数据
    def dau_datasets(self, num=10):
        self.init_dir()
        self.run("train", num=num)
        self.run("test", num=num)

    def run(self, prefix, num=10):
        for i in range(num):  # 扩增10倍
            img_list = []
            label_list = []
            if prefix == "train":
                data = zip(self.x_train, self.y_train)
            else:
                data = zip(self.x_test, self.y_test)
            for x, y in tqdm(data):
                img = self.gen.random_transform(x, seed=None)
                img_list.append(img)
                label_list.append(y)
            xs = np.array(img_list)
            ys = np.array(label_list)
            np.save((self.x_path + "_{}").format(prefix, i), xs)
            np.save((self.y_path + "_{}").format(prefix, i), ys)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # 初始化参数
    params = {
        "data_name": None,
        "width": None,
        "height": None,
        "channel": None,
        "base_dir": None,
        "model_name": None
    }

    params["data_name"] = "svhn"
    params["width"] = 32
    params["height"] = 32
    params["channel"] = 3
    params["base_dir"] = "dau/{}_harder".format("svhn")
    daugor(params).dau_datasets()
