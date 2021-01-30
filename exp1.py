# 进行随机sample试验
# d1,标签均匀性
# d2,标签相似性
# d3.拓扑分析
import datetime
import json
import os
import random
import time

import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from neural_cov import CovInit

plt.switch_backend('agg')
import numpy as np
import pandas as pd
from keras import backend as K
from keras.engine.saving import load_model

import cov_config
import data_gener
import metrics
import model_conf
import deepdiv
from utils import train_model, shuffle_data


###
# 工具函数
#

# 输出配置信息
def param2txt(file_path, msg):
    f = open(file_path, "w")
    f.write(msg)
    f.close


# 创建一个实验文件夹
def mk_exp_dir(params):
    # 6.进行试验
    ## 6.1 创建文件夹并储存该次参数文件
    base_path = "./result"
    pair_name = model_conf.get_pair_name(params["data_name"], params["model_name"])
    dir_name = datetime.datetime.now().strftime("%m%d%H%M") + "_" + exp_name + "_" + pair_name
    txt_name = pair_name + ".txt"
    base_path = base_path + "/" + dir_name
    if not os.path.exists(base_path):
        os.mkdir(base_path)
    txt_path = base_path + "/" + txt_name
    param2txt(txt_path, json.dumps(params, indent=1))

    return base_path


def add_df(df, csv_data):
    if df is None:  # 如果是空的
        df = pd.DataFrame(csv_data, index=[0])
    else:
        df.loc[df.shape[0]] = csv_data
    return df


###
# 准备数据
#  训练集 5w不变  测试集1w
#  从测试集里  2000  5000 ...
#  观察该次测试里  指标的覆盖率  和  发现的错误用例数量
#

def get_select_data(x_ori_test, y_ori_test, x_dau_test, y_dau_test, len_select_dau, len_select_ori, seed, shuffle_seed):
    x_dau_select, _, y_dau_select, _ = train_test_split(x_dau_test, y_dau_test, train_size=len_select_dau,
                                                        random_state=seed)
    x_ori_select, _, y_ori_select, _ = train_test_split(x_ori_test, y_ori_test, train_size=len_select_ori,
                                                        random_state=seed)

    x_select = np.concatenate([x_dau_select, x_ori_select], axis=0)  # 选择集
    y_select = np.concatenate([y_dau_select, y_ori_select], axis=0)

    np.random.seed(shuffle_seed)
    shuffle_indices = np.random.permutation(len(x_select))
    x_s = x_select[shuffle_indices]  # 混洗,每次混洗要不一样,并选出前len的数据
    y_s = y_select[shuffle_indices]
    return x_s, y_s


###
# 指标数据
#
def get_div_exp_data(x_s, y_s, ori_model, base_path, is_anlyze=False):
    csv_data = {}
    s = time.time()
    div_c, div_v = deepdiv.cal_d_v(x_s, y_s, params["nb_classes"], ori_model, base_path=base_path, is_anlyze=is_anlyze)
    e = time.time()
    csv_data["div"] = div_c
    csv_data["var"] = div_v
    csv_data["div_time_cost"] = e - s
    return csv_data


def get_cov_exp_data(x_s, y_s, cov_initer, suffix=""):
    csv_data = {}
    cov_nac, cov_nbc, cov_snac, cov_kmnc, cov_tknc, cov_lsc, cov_dsc = None, None, None, None, None, None, None,

    input_layer = cov_initer.get_input_layer()
    layers = cov_initer.get_layers()

    ss = time.time()
    nac = metrics.nac(x_s, input_layer, layers, t=0.75)
    cov_nac = nac.fit()
    ee = time.time()
    csv_data["nac_time{}".format(suffix)] = ee - ss

    ss = time.time()
    nbc = cov_initer.get_nbc()
    cov_nbc = nbc.fit(x_s, use_lower=True)
    cov_snac = nbc.fit(x_s, use_lower=False)
    ee = time.time()
    csv_data["nbc_time{}".format(suffix)] = ee - ss

    ss = time.time()
    kmnc = cov_initer.get_kmnc()
    cov_kmnc = kmnc.fit(x_s)
    ee = time.time()
    csv_data["kmnc_time{}".format(suffix)] = ee - ss

    ss = time.time()
    tknc = metrics.tknc(x_s, input_layer, layers, k=1)
    cov_tknc = tknc.fit(list(range(len(x_s))))
    ee = time.time()
    csv_data["tknc_time{}".format(suffix)] = ee - ss

    ss = time.time()
    lsc = cov_initer.get_lsc(k_bins=1000, index=-1)
    cov_lsc = lsc.fit(x_s, y_s)
    ee = time.time()
    csv_data["lsc_time{}".format(suffix)] = ee - ss
    # ss = time.time()
    # dsc = cov_initer.get_dsc(k_bins=1000, )
    # cov_dsc = dsc.fit(x_s, y_s)
    # ee = time.time()
    #
    # csv_data["dsc_time{}".format(suffix)] = ee - ss

    csv_data["cov_nac{}".format(suffix)] = cov_nac
    csv_data["cov_nbc{}".format(suffix)] = cov_nbc
    csv_data["cov_snac{}".format(suffix)] = cov_snac
    csv_data["cov_tknc{}".format(suffix)] = cov_tknc
    csv_data["cov_kmnc{}".format(suffix)] = cov_kmnc
    csv_data["cov_lsc{}".format(suffix)] = cov_lsc
    # csv_data["cov_dsc{}".format(suffix)] = cov_dsc
    return csv_data


# 重训练模型
def retrain_model(ori_model, x_s, y_s_vec, x_val, y_val_vec, ):
    temp_path = model_conf.get_temp_model_path(params["data_name"], params["model_name"], "temp")
    if not os.path.exists(temp_path):
        os.makedirs(temp_path)
    new_model_name = exp_name + str(len(x_s)) + ".hdf5"
    filepath = "{}/{}".format(temp_path, new_model_name)
    trained_model = train_model(ori_model, filepath, x_s, y_s_vec, x_val, y_val_vec)
    return trained_model


###
# 实验
#
def exp_detail(ori_model, cov_initer, ys_psedu, seed, x_s, y_s, base_path, is_cal_cov=True, is_cal_div=True,
               is_anlyze=False):
    right_num = np.sum(y_s == ys_psedu)
    wrong_num = len(y_s) - right_num

    # 结果行数据
    csv_data = {}
    csv_data["seed"] = seed
    csv_data["data_size"] = len(x_s)
    csv_data["wrong_num"] = wrong_num
    csv_data["right_num"] = right_num

    if is_cal_div:
        # div实验
        exp_div_data = get_div_exp_data(x_s, ys_psedu, ori_model, base_path, is_anlyze=is_anlyze)
        csv_data = dict(csv_data, **exp_div_data)

    # cov实验
    if is_cal_cov:
        # 4.计算整体的众多cov
        exp_cov_data = get_cov_exp_data(x_s, ys_psedu, cov_initer)
        csv_data = dict(csv_data, **exp_cov_data)
    del x_s
    del y_s
    return csv_data


#  扩增测试数据合并原始验证数据
def prepare_data():
    # 1.原始数据
    X_train, X_test, Y_train_vec, Y_test_vec = data_gener.gen_ori_data(params["data_name"])
    Y_train = np.argmax(Y_train_vec, axis=1)
    Y_test = np.argmax(Y_test_vec, axis=1)
    len_test = len(X_test)

    # 2.混洗
    np.random.seed(42)  # 指定随机数种子,保证每次混洗后的数据集都一样
    shuffle_indices = np.random.permutation(np.arange(len(X_train)))
    X_train, Y_train = X_train[shuffle_indices], Y_train[shuffle_indices]

    # 3.这里面只有前5w个是训练集
    x_train, y_train = X_train[len_test:], Y_train[len_test:]  # 最终的训练集 5w

    # 获取扩增测试集
    Xa_train, Xa_test, Ya_train, Ya_test = data_gener.gen_dau_data(params["dau_name"], use_norm=True,
                                                                   num=dau_num)

    # 扩增测试集 合并 原始测试集 2w
    x_test = np.concatenate([Xa_test, X_test], axis=0)
    y_test = np.concatenate([Ya_test, Y_test], axis=0)
    return x_train, x_test, y_train, y_test


# 对外暴露一个方法,获得测试集的数量
def get_exp1_test_size(data_name, dau_name):
    global params
    params = {}
    params["data_name"] = data_name
    params["dau_name"] = dau_name
    _, x_test, _, _ = prepare_data()
    return len(x_test)


# 对外暴露一个方法,获得选择集的总错误数
def get_exp1_fault_size(data_name, dau_name, model_name):
    global params
    params = {}
    params["data_name"] = data_name
    params["dau_name"] = dau_name
    params["model_name"] = model_name
    X_train, X_test, Y_train, Y_test = prepare_data()  # 原始数据
    ori_model = load_model(model_conf.get_model_path(params["data_name"], params["model_name"]))  # 模型
    prob_matrixc = ori_model.predict(X_test)
    ys_psedu = np.argmax(prob_matrixc, axis=1)  # 每行最大的置信度作为伪标签,充分性分析使用伪标签,重训练使用正确标签
    right_num = np.sum(Y_test == ys_psedu)
    wrong_num = len(Y_test) - right_num
    return wrong_num


#  扩增测试数据合并原始验证数据
#  加入验证集,让测试集变得更大 共4w
def prepare_data4():
    # 1.原始数据
    X_train, X_test, Y_train_vec, Y_test_vec = data_gener.gen_ori_data(params["data_name"])
    Y_train = np.argmax(Y_train_vec, axis=1)
    Y_test = np.argmax(Y_test_vec, axis=1)
    len_test = len(X_test)

    # 2.混洗
    np.random.seed(42)  # 指定随机数种子,保证每次混洗后的数据集都一样
    shuffle_indices = np.random.permutation(np.arange(len(X_train)))
    X_train, Y_train = X_train[shuffle_indices], Y_train[shuffle_indices]

    # 3.这里面只有前5w个是训练集
    x_train, y_train = X_train[len_test:], Y_train[len_test:]  # 最终的训练集 5w

    # 选择集
    X_ori_select, Y_ori_select = X_train[:len_test], Y_train[:len_test]

    # 获取扩增测试集
    _, Xa_test, _, Ya_test = data_gener.gen_dau_data(params["dau_name"], use_norm=True,
                                                     num=2)

    # 扩增测试集 2w 合并 原始测试集 1w 合并 原始选择集 1w  =4w
    x_test = np.concatenate([Xa_test, X_test, X_ori_select], axis=0)
    y_test = np.concatenate([Ya_test, Y_test, Y_ori_select], axis=0)
    return x_train, x_test, y_train, y_test


def print_info(right_num, wrong_num, len_X_test, len_Y_test, len_X_train, len_Y_train):
    #### 打印消息
    print("train", len_X_train, len_Y_train)
    print("test", len_X_test, len_Y_test)
    print("right_num", right_num)
    print("wrong_num", wrong_num)


def exp(params):
    K.clear_session()

    # 参数配置
    exp_params = {
        "b_list": [0],
        "ub_list": [0.99],
        "k_list": [0.1, 0.3],  # size(原始测试 )* k = size(子集)
        "sample_num": 30  # 每个子集的采样次数
    }
    # 0. 初始化参数
    sample_num = exp_params["sample_num"]
    k_list = exp_params["k_list"]
    b_list, ub_list = exp_params["b_list"], exp_params["ub_list"]  # 不分块  只选取高置信度部分,低的不要 # > 0.22
    b, ub = b_list[0], ub_list[0]
    cov_config.boundary = b  # 设置边界
    cov_config.up_boundary = ub
    is_cal_cov = True  # 控制
    is_cal_div = True
    base_path = mk_exp_dir(dict(params, **exp_params))  # 创建文件夹

    # 1. 获取原始数据
    X_train, X_test, Y_train, Y_test = prepare_data()  # 原始数据
    ori_model = load_model(model_conf.get_model_path(params["data_name"], params["model_name"]))  # 模型
    prob_matrixc = ori_model.predict(X_test)
    ys_psedu = np.argmax(prob_matrixc, axis=1)  # 每行最大的置信度作为伪标签,充分性分析使用伪标签,重训练使用正确标签
    right_num = np.sum(Y_test == ys_psedu)
    wrong_num = len(Y_test) - right_num
    # 打印信息
    print_info(right_num, wrong_num, len(X_test), len(Y_test), len(X_train), len(Y_train))

    # 2. 初始化覆盖指标
    cov_initer = CovInit(X_train, Y_train, params)

    # 4.实验
    for k in k_list:
        # 3. 随机选取数据种子
        random.seed(42)  # 固定下列随机数
        seed_list = random.sample(range(0, 1000), 500)
        seed_point = 0
        size = int(len(X_test) * k)
        df = None
        csv_path = base_path + "/exp1_{}.csv".format(size, )
        # 构建相同size不同ratio数据集
        for i in range(sample_num):
            seed = seed_list[seed_point]
            X, Y = shuffle_data(X_test, Y_test, seed)
            x_s, y_s = X[:size], Y[:size]  # 每次都随机选择出1000条
            prob_matrixc = ori_model.predict(x_s)
            ys_psedu = np.argmax(prob_matrixc, axis=1)  # 每行最大的置信度作为伪标签,充分性分析使用伪标签,重训练使用正确标签
            seed_point += 1  # 改变随机数
            csv_data = exp_detail(ori_model, cov_initer, ys_psedu, seed, x_s, y_s, base_path,
                                  is_cal_cov=is_cal_cov, is_cal_div=is_cal_div, )
            df = add_df(df, csv_data)
            df.to_csv(csv_path, index=False)
        analyze(size, base_path=base_path)


def analyze(data_size, base_path=None):
    from scipy.stats import pearsonr, spearmanr
    import seaborn
    if base_path is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    csv_path = base_path + "/exp1_{}.csv".format(data_size)  # 3k 6k 3w 6w
    fig_dir = base_path + "/fig"
    os.makedirs(fig_dir, exist_ok=True)
    fig_path = "{}/{}_{}_{}_all.png".format(fig_dir, params["data_name"], params["model_name"], data_size)
    df = pd.read_csv(csv_path)
    div_col_arr = ["div", "var", ]
    cov_col_arr = ["cov_nac", "cov_nbc", "cov_snac", "cov_tknc",
                   "cov_lsc", "cov_kmnc", ]  #
    # cov_col_arr = ["cov_dsc", ]
    col_arr = div_col_arr + cov_col_arr
    row = 2
    cr = 4
    del col_arr[1]
    fig, axes = plt.subplots(row, cr, figsize=(20, 10))
    fig.suptitle("{} datasize = {} ".format(params["data_name"], df["data_size"][0]))

    for i in range(len(col_arr)):
        col = col_arr[i]
        # x, y = df[col], df["acc_imp"]
        x, y = df[col], df["wrong_num"]
        p_res = pearsonr(x, y)
        sp_res = spearmanr(x, y)
        r1, p1 = p_res
        r2, p2 = sp_res
        if i >= cr:
            xi = 1
            yi = i - cr
        else:
            xi = 0
            yi = i
        axes[xi][yi].set_title("col {} pr r={} p={} sp r={} p={} ".format(col, np.round(r1, 2), np.round(p1, 2),
                                                                          np.round(r2, 2), np.round(p2, 2)))
        ax = seaborn.jointplot(col, "wrong_num", data=df, kind='reg', stat_func=None, color=None, ax=axes[xi][yi])
    fig.savefig(fig_path)
    plt.close()
    print(fig_path)


dau_num = 1
exp_name = "exp1"
params = None  # 待赋值


def run(p, gpu_no=0, ename=None):
    global params
    params = p
    if ename is not None:
        global exp_name
        exp_name = ename
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_no)
    exp(params)


if __name__ == '__main__':
    pass

    params = {
        "model_name": model_conf.LeNet1,  # 选取模型
        "data_name": model_conf.mnist,  # 选取数据集
        "nb_classes": model_conf.fig_nb_classes,  # 数据集分类数
        "dau_name": "mnist_harder"  # 选取扩增数据集名称
    }
    run(params, 0)
