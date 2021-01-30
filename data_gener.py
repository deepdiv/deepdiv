import random

import numpy as np
from keras.datasets import mnist, fashion_mnist, cifar10
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

import SVNH_DatasetUtil
# 根据标签筛选数据
import model_conf


def get_data_by_label(X, Y, label, ):
    idx_arr = np.where(Y == label)
    y = Y[idx_arr]
    x = X[idx_arr]
    return x, y


# 带所选下标
def get_data_by_label_with_idx(X, Y, label, ):
    idx_arr = np.where(Y == label)
    y = Y[idx_arr]
    x = X[idx_arr]
    return x, y, idx_arr[0]


def remove_data_by_label(X, Y, label, ):
    idx_arr = np.where(Y != label)
    y = Y[idx_arr]
    x = X[idx_arr]
    return x, y


def gen_data_mnist(nb_classes=10, label=None):
    # 加载数据集
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()  # 28*28
    if label is not None:  # 根据标签,获取数据
        X_test, Y_test = get_data_by_label(X_test, Y_test, label)
        X_train, Y_train = get_data_by_label(X_train, Y_train, label)
    X_train = X_train.astype('float32').reshape(-1, 28, 28, 1)
    X_test = X_test.astype('float32').reshape(-1, 28, 28, 1)
    X_train /= 255
    X_test /= 255
    # 数据预处理
    Y_train = np_utils.to_categorical(Y_train, nb_classes)
    Y_test = np_utils.to_categorical(Y_test, nb_classes)
    return X_train, X_test, Y_train, Y_test,


def gen_data_fashion(nb_classes=10, label=None):
    # 加载数据集
    (X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()  # 28*28
    if label is not None:  # 根据标签,获取数据
        X_test, Y_test = get_data_by_label(X_test, Y_test, label)
        X_train, Y_train = get_data_by_label(X_train, Y_train, label)
    X_train = X_train.astype('float32').reshape(-1, 28, 28, 1)
    X_test = X_test.astype('float32').reshape(-1, 28, 28, 1)
    X_train /= 255
    X_test /= 255
    # 数据预处理
    Y_train = np_utils.to_categorical(Y_train, nb_classes)
    Y_test = np_utils.to_categorical(Y_test, nb_classes)
    return X_train, X_test, Y_train, Y_test,


def gen_data_cifar(nb_classes=10, label=None):
    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()  # 28*28
    if label is not None:  # 根据标签,获取数据
        X_test, Y_test = get_data_by_label(X_test, Y_test, label)
        X_train, Y_train = get_data_by_label(X_train, Y_train, label)
    X_train = X_train.astype('float32').reshape(-1, 32, 32, 3)
    X_test = X_test.astype('float32').reshape(-1, 32, 32, 3)
    X_train /= 255
    X_test /= 255
    Y_train = Y_train.reshape(-1)
    Y_test = Y_test.reshape(-1)

    Y_train = np_utils.to_categorical(Y_train, nb_classes)
    Y_test = np_utils.to_categorical(Y_test, nb_classes)
    return X_train, X_test, Y_train, Y_test,


def gen_data_svhn(nb_classes=10, label=None):
    (X_train, Y_train), (X_test, Y_test) = SVNH_DatasetUtil.load_data()  # 32*32
    Y_test = np.argmax(Y_test, axis=1)
    Y_train = np.argmax(Y_train, axis=1)
    if label is not None:  # 根据标签,获取数据
        X_test, Y_test = get_data_by_label(X_test, Y_test, label)
        X_train, Y_train = get_data_by_label(X_train, Y_train, label)
    Y_train = np_utils.to_categorical(Y_train, nb_classes)
    Y_test = np_utils.to_categorical(Y_test, nb_classes)
    return X_train, X_test, Y_train, Y_test,


##################
# 获得原始数据
##################
def gen_ori_data(data_name, label=None):
    X_train, X_test, Y_train, Y_test = None, None, None, None,
    if data_name.startswith("mnist"):
        X_train, X_test, Y_train, Y_test, = gen_data_mnist(label=label)  # 获得原始数据集
    elif data_name.startswith("fashion"):
        X_train, X_test, Y_train, Y_test, = gen_data_fashion(label=label)  # 获得原始数据集
    elif data_name.startswith("svhn"):
        X_train, X_test, Y_train, Y_test, = gen_data_svhn(label=label)  # 获得原始数据集
    elif data_name.startswith("cifar"):
        X_train, X_test, Y_train, Y_test, = gen_data_cifar(label=label)  # 获得原始数据集
    elif data_name.startswith("Drebin"):
        X_train, X_test, Y_train, Y_test, = gen_Drebin_data()
    return X_train, X_test, Y_train, Y_test


##################
# 获得扩增数据
##################
def gen_dau_data(data_name, use_norm=False, num=10):
    if num is None:
        num = 10

    def gen_dau_data_detail(data_name, pre):
        x_arr = []
        y_arr = []
        for i in range(num):
            x_path = "./dau/{}/x_{}_{}.npy".format(data_name, pre, i)
            y_path = "./dau/{}/y_{}_{}.npy".format(data_name, pre, i)
            x = np.load(x_path)
            y = np.load(y_path)
            x_arr.append(x)
            y_arr.append(y)
        x_dau = np.concatenate(x_arr, axis=0)
        y_dau = np.concatenate(y_arr, axis=0)
        return x_dau, y_dau

    X_train, Y_train = gen_dau_data_detail(data_name, "train")
    X_test, Y_test = gen_dau_data_detail(data_name, "test")
    if use_norm:
        X_train /= 255
        X_test /= 255
    if data_name == "pdf":
        pass
    Y_train = Y_train.flatten()
    Y_test = Y_test.flatten()  # cifar扩充后是(-1,1)
    return X_train, X_test, Y_train, Y_test


def gen_dau_data_op(data_name, num=10):
    # print(data_name)

    def gen_dau_data_op_detail(pre):
        op_arr = []
        for i in range(num):
            op_path = "./dau/{}/op_{}_{}.npy".format(data_name, pre, i)
            op = np.load(op_path)
            op_arr.append(op)
        x_op = np.concatenate(op_arr, axis=0)
        return x_op

    x_op_train = gen_dau_data_op_detail("train")
    x_op_test = gen_dau_data_op_detail("test")
    return x_op_train, x_op_test


##################
# 分割扩增数据
##################
def split_dau_data(X, Y, nb_class, size=0.5, random_state=42, ):
    X_train_list, Y_train_list, = [], []
    X_test_list, Y_test_list, = [], []

    for i in range(nb_class):
        # 获取每个lable下的所有数据
        X_i, Y_i = get_data_by_label(X, Y, i)
        data_size = len(X_i)
        # 均匀分成两份
        # 生成随机序列混洗
        np.random.seed(random_state)
        shuffle_indices = np.random.permutation(np.arange(data_size))
        train_values_shuffled = X[shuffle_indices]
        train_labels_shuffled = Y[shuffle_indices]

        # 均匀分割
        Xa_train = train_values_shuffled[:int(data_size * size)]
        Xa_test = train_values_shuffled[int(data_size * size):]
        Ya_train = train_labels_shuffled[:int(data_size * size)]
        Ya_test = train_labels_shuffled[int(data_size * size):]
        X_train_list.append(Xa_train)
        Y_train_list.append(Ya_train)
        X_test_list.append(Xa_test)
        Y_test_list.append(Ya_test)

    X_train = np.concatenate(X_train_list, axis=0)
    Y_train = np.concatenate(Y_train_list, axis=0)
    X_test = np.concatenate(X_test_list, axis=0)
    Y_test = np.concatenate(Y_test_list, axis=0)
    # print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)
    # 随机分成两份
    # Xa_train, Xa_test, Ya_train, Ya_test = train_test_split(X_test, Y_test, train_size=size, random_state=random_state)

    return X_train, X_test, Y_train, Y_test


##################
# uniform_sampling 对数据集中每个class 随机选取同等数量的数据
##################
def uniform_sampling(test_size, X, Y, nb_class, random_state):
    ori_size = int(test_size / nb_class)  # 每个lable选取ori_size个
    # ori_size = 1 / nb_class
    X_list, Y_list, = [], []
    for i in range(nb_class):
        # 获取每个lable下的所有数据
        X_i, Y_i = get_data_by_label(X, Y, i)
        # 每个label选出ori_size个数据
        if len(X_i) == ori_size:
            X_usi, Y_usi = X_i, Y_i
        else:
            X_usi, _, Y_usi, _ = train_test_split(X_i, Y_i, train_size=ori_size, random_state=random_state)
        # print(i, len(X_usi))
        X_list.append(X_usi)
        Y_list.append(Y_usi)
    X_us = np.concatenate(X_list, axis=0)
    Y_us = np.concatenate(Y_list, axis=0)
    # 混洗
    np.random.seed(random_state)
    shuffle_indices = np.random.permutation(len(X_us))
    X_us = X_us[shuffle_indices]
    Y_us = Y_us[shuffle_indices]
    return X_us, Y_us


##################
# Non-uniform_sampling 选中的class取0.5的数据, 其他class 随机选取同等数量的数据
##################
def non_uniform_sampling(test_size, X, Y, nb_class, random_state, ratio=0.5):
    X_select, Y_select = None, None
    ori_size = int(test_size * ratio)
    other_size = test_size - ori_size
    # other_size = int(test_size * (1 - ratio))
    X_list, Y_list, = [], []
    while True:
        random.seed(random_state)  # 固定下列随机数
        select_lable = random.randint(0, nb_class - 1)  # 随机选择一个lable采样50%
        # 获取select_lable下的所有数据
        X_i, Y_i = get_data_by_label(X, Y, select_lable)
        if len(X_i) > ori_size:
            # 数据总量比 要选的数据多
            X_select, _, Y_select, _ = train_test_split(X_i, Y_i, train_size=ori_size, random_state=random_state)
            print("select", select_lable)
            break
        elif len(X_i) == ori_size:
            # 数据总量和要选的数据一样多
            X_select, Y_select, = X_i, Y_i
            print("select all", select_lable)
        else:
            # 数据总量比要选的数据少
            random_state += 1
    # 选择ori_size个数据
    X_list.append(X_select)
    Y_list.append(Y_select)
    # print("select", len(X_select), np.unique(Y_select))
    # 获取除select_lable以外的所有数据
    X_r, Y_r = remove_data_by_label(X, Y, select_lable)
    # 从中选择剩余的50%数据
    X_others, _, Y_others, _ = train_test_split(X_r, Y_r, train_size=other_size, random_state=random_state)
    X_list.append(X_others)
    Y_list.append(Y_others)
    # print("others", len(X_others), np.unique(Y_others))
    X_ns = np.concatenate(X_list, axis=0)
    Y_ns = np.concatenate(Y_list, axis=0)
    return X_ns, Y_ns


##################
# random_sampling  不按class随机选数据
##################
def random_sampling(test_size, X, Y, nb_class, random_state):
    if len(X) == test_size:
        X_rsi, Y_rsi = X, Y,
    else:
        X_rsi, _, Y_rsi, _ = train_test_split(X, Y, train_size=test_size, random_state=random_state)
    return X_rsi, Y_rsi


def gen_data(data_name, label=None, use_dau=False, nb_classes=None):
    X_train, X_val, X_test, Y_train, Y_val, Y_test = None, None, None, None, None, None
    if use_dau and data_name != "pdf":
        # print("label", label)
        x_path = "./dau/{}_xs.npy".format(data_name)
        y_path = "./dau/{}_ys.npy".format(data_name)
        X_train, Y_train = None, None
        X_test = np.load(x_path)
        Y_test = np.load(y_path)
        # 数据预处理
        if data_name in [model_conf.mnist, model_conf.cifar10, model_conf.fashion]:
            X_test /= 255
        Y_test = np_utils.to_categorical(Y_test, nb_classes)
        # 格式问题修正
        # temp = []
        # for i in range(len(X_test)):
        #     t = X_test[i].reshape(32, 32, 3)
        #     temp.append(t)
        # np.save(x_path, np.array(temp))
        # print(np.load(x_path).shape)
    else:
        if data_name.startswith("mnist"):
            X_train, X_test, Y_train, Y_test, = gen_data_mnist(label=label)  # 获得原始数据集
        elif data_name.startswith("fashion"):
            X_train, X_test, Y_train, Y_test, = gen_data_fashion(label=label)  # 获得原始数据集
        elif data_name.startswith("svhn"):
            X_train, X_test, Y_train, Y_test, = gen_data_svhn(label=label)  # 获得原始数据集
        elif data_name.startswith("cifar"):
            X_train, X_test, Y_train, Y_test, = gen_data_cifar(label=label)  # 获得原始数据集
        elif data_name.startswith("Drebin"):
            X_train, X_test, Y_train, Y_test, = gen_Drebin_data()

    # attack_lst = ['cw', 'fgsm', 'jsma', 'bim']

    # adv_image_test_arr = []
    # adv_label_test_arr = []
    # for attack in attack_lst:
    #     im, lab = model_conf.get_adv_path(attack, name, model_name)
    #     attack_test = np.load(im)
    #     attack_lable = np.load(lab)
    #     adv_image_test_arr.append(attack_test)
    #     adv_label_test_arr.append(attack_lable)
    #
    # adv_X_test = np.concatenate(adv_image_test_arr, axis=0)
    # adv_Y_test = np.concatenate(adv_label_test_arr, axis=0)
    #
    # adv_Y_test = np_utils.to_categorical(adv_Y_test, nb_classes)
    # X_test = adv_X_test
    # Y_test = adv_Y_test

    # 1w条数据
    # adv_X_test, _, adv_Y_test, _ = train_test_split(adv_X_test, adv_Y_test, train_size=5000, random_state=42)
    # X_test, _, Y_test, _ = train_test_split(X_test, Y_test, train_size=5000, random_state=42)
    # X_test = np.r_[X_test, adv_X_test]  # 共1w条
    # Y_test = np.r_[Y_test, adv_Y_test]

    # adv_X_test, _, adv_Y_test, _ = train_test_split(adv_X_test, adv_Y_test, train_size=5000)
    # X_test, _, Y_test, _ = train_test_split(X_test, Y_test, train_size=5000)
    # adv_Y_test = np_utils.to_categorical(adv_Y_test, nb_classes)
    # X_test = np.concatenate([X_test, adv_X_test], axis=0)  # 共1w条 2000:8000
    # Y_test = np.concatenate([Y_test, adv_Y_test], axis=0)
    return X_train, X_test, Y_train, Y_test


def gen_adv(data_name, model_name, nb_classes=10, label=None, attack_lst=None):
    if attack_lst is None:
        attack_lst = ['cw', 'fgsm', 'jsma', 'bim']
    print(attack_lst)
    adv_image_test_arr = []
    adv_label_test_arr = []
    for attack in attack_lst:
        im, lab = model_conf.get_adv_path(attack, data_name, model_name)
        attack_test = np.load(im)
        attack_lable = np.load(lab)
        adv_image_test_arr.append(attack_test)
        adv_label_test_arr.append(attack_lable)
    adv_X_test = np.concatenate(adv_image_test_arr, axis=0)
    adv_Y_test = np.concatenate(adv_label_test_arr, axis=0)
    if label is not None:  # 根据标签,获取数据
        adv_X_test, adv_Y_test = get_data_by_label(adv_X_test, adv_Y_test, label)
    adv_Y_test = np_utils.to_categorical(adv_Y_test, nb_classes)
    return adv_X_test, adv_Y_test


#
def gen_data_and_adv(data_name, model_name, nb_classes=10, adv_size=None, ori_size=None, label=None, use_adv=False,
                     attack_lst=None, random_state=42, use_dau=False):
    X_train, X_test, Y_train, Y_test = gen_data(data_name, label=label, use_dau=use_dau, nb_classes=nb_classes)
    print("数据总量", "X_test", len(X_test))
    if ori_size is not None:
        X_test, _, Y_test, _ = train_test_split(X_test, Y_test, train_size=ori_size, random_state=random_state)
        print("选取数据量", "X_test", len(X_test))
    if use_adv:
        adv_X_test, adv_Y_test = gen_adv(data_name, model_name, nb_classes=nb_classes, label=label,
                                         attack_lst=attack_lst)
        # print("数据总量", "adv_X_test", len(adv_X_test), )
        if adv_size is not None:
            adv_X_test, _, adv_Y_test, _ = train_test_split(adv_X_test, adv_Y_test, train_size=adv_size,
                                                            random_state=random_state)
        print("选取数据量", "adv_X_test", len(adv_X_test), )
        X_test = np.concatenate([X_test, adv_X_test], axis=0)
        Y_test = np.concatenate([Y_test, adv_Y_test], axis=0)
    return X_train, X_test, Y_train, Y_test


def gen_Drebin_data():
    X_train = np.load("./Drebin/training_xs.npy")
    Y_train = np.load("./Drebin/training_ys.npy")
    X_test = np.load("./Drebin/testing_xs.npy")
    Y_test = np.load("./Drebin/testing_ys.npy")
    print(X_test.shape)
    print(Y_test.shape)
    return X_train, X_test, Y_train, Y_test

# # 获得数据扩增后的数据
# def gen_dau_data():
#     X_train = np.load("./Drebin/training_xs.npy")
#     Y_train = np.load("./Drebin/training_ys.npy")
#     X_test = np.load("./Drebin/testing_xs.npy")
#     Y_test = np.load("./Drebin/testing_ys.npy")

# if __name__ == '__main__':
#     x_path = "./dau/{}/xs_1.npy".format("mnist")
#     y_path = "./dau/{}/ys_1.npy".format("mnist")
# X_train, Y_train = None, None
# X_test = np.load(x_path)
# Y_test = np.load(y_path)
# print(Y_test.shape, X_test.shape)
# gen_data(model_conf.mnist, use_dau=True, nb_classes=10)
# gen_data(model_conf.cifar10, use_dau=True, nb_classes=10)
# gen_data(model_conf.fashion, use_dau=True, nb_classes=10)
# gen_data(model_conf.svhn, use_dau=True, nb_classes=10)
