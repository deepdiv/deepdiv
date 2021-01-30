# 进行随机sample试验
# d1,标签均匀性
# d2,标签相似性
# d3.拓扑分析
import datetime
import functools
import json
import os
import random
import time

import keras
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

import cov_config
import deepdiv
from gen_model.gen_model_utils import gen_exp2_model
from neural_cov import neural_cov

plt.switch_backend('agg')
import numpy as np
from keras.engine.saving import load_model

import data_gener
import model_conf
from utils import train_model
from keras import backend as K


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
    os.makedirs(base_path + "/ps_data", exist_ok=True)
    txt_path = base_path + "/" + txt_name
    param2txt(txt_path, json.dumps(params, indent=1))
    return base_path


def retrain_model(params, ori_model, x_si, y_si_vector, Xa_test, Ya_test_vec, smaple_method, idx=0, verbose=True):
    temp_path = model_conf.get_temp_model_path(params["data_name"], params["model_name"], smaple_method)
    if not os.path.exists(temp_path):
        os.makedirs(temp_path)
    new_model_name = exp_name + str(idx) + "_.hdf5"
    filepath = "{}/{}".format(temp_path, new_model_name)
    trained_model = train_model(ori_model, filepath, x_si, y_si_vector, Xa_test, Ya_test_vec, verbose=verbose)
    return trained_model


##################################################################
# 准备优先级序列
###################################


def prepare_data():
    # 1. 获得原始数据
    X_train, X_test, Y_train_vec, Y_test_vec = data_gener.gen_ori_data(params["data_name"])
    Y_train = np.argmax(Y_train_vec, axis=1)
    Y_test = np.argmax(Y_test_vec, axis=1)
    len_test = len(X_test)
    # 2. 获得扩增数据
    Xa_train, _, Ya_train, _ = data_gener.gen_dau_data(params["dau_name"], use_norm=True, num=train_dau_num)
    _, Xa_test, _, Ya_test = data_gener.gen_dau_data(params["dau_name"], use_norm=True, num=test_dau_num)
    # 2. 混洗训练集,混洗扩增数据:
    np.random.seed(42)
    shuffle_indices = np.random.permutation(np.arange(len(X_train)))
    X_train, Y_train = X_train[shuffle_indices], Y_train[shuffle_indices]
    Xa_train, Ya_train = Xa_train[shuffle_indices], Ya_train[shuffle_indices]
    # 3. 拆分训练集,拆分扩增数据
    X_ori_select, Y_ori_select = X_train[:len_test], Y_train[:len_test]
    Xa_select, Ya_select = Xa_train[:len_test], Ya_train[:len_test]
    x_train, y_train = X_train[len_test:], Y_train[len_test:]  # 最终的训练集
    # 4. 按比例选出部分扩增集
    Xa_add, _, Ya_add, _ = train_test_split(Xa_select, Ya_select, train_size=0.3, random_state=42)
    # 4. 混合原始选择集与扩增选择集
    x_select = np.concatenate([X_ori_select, Xa_add], axis=0)
    y_select = np.concatenate([Y_ori_select, Ya_add], axis=0)
    # 混洗选择集
    np.random.seed(42)
    shuffle_indices = np.random.permutation(len(x_select))
    x_select = x_select[shuffle_indices]  # 混洗,每次混洗要不一样,并选出前len的数据
    y_select = y_select[shuffle_indices]

    # 5.测试集不变
    x_test, y_test = X_test, Y_test
    # 6.验证集为测试集的扩增集
    x_val, y_val = Xa_test, Ya_test
    return x_train, x_test, y_train, y_test, x_select, y_select, x_val, y_val


def prepare_model():
    X_train, X_test, Y_train, Y_test, X_select, Y_select, X_val, Y_val = prepare_data()
    print("train", len(X_train), len(Y_train))
    print("test", len(X_test), len(Y_test))
    print("select", len(X_select), len(Y_select))
    print("val", len(X_val), len(Y_val))
    gen_exp2_model(params["data_name"], params["model_name"], X_train, Y_train, X_test, Y_test, )


# 本次使用的覆盖方法
def get_cov_name_and_func(cover, max_select_size, use_max=False):
    if use_max:
        cal_kmnc_cov_limit = functools.partial(cover.cal_kmnc_cov, max_select_size=max_select_size, time_limit=3600)
    else:
        cal_kmnc_cov_limit = functools.partial(cover.cal_kmnc_cov, time_limit=3600)
    cov_name_list = ["KMNC"]  # "DSC",
    func_list = [cal_kmnc_cov_limit]
    return cov_name_list, func_list


# 覆盖方法对应的优先级顺序
def get_cov_data(base_path, cov_name_list, func_list, df, ):
    csv_data = get_ps_csv_data()
    for name, func in zip(cov_name_list, func_list):
        rate, t_collection, rank_lst, t_selection_cam, rank_lst2, t_selection_ctm = func()
        csv_data["name"] = name
        csv_data["t_collection"] = t_collection
        if rank_lst is not None:
            save_path = base_path + "/ps_data/{}_{}_rank_list.npy".format(name, "cam")
            np.save(save_path, rank_lst)
            csv_data["cam_t_selection"] = t_selection_cam
            csv_data["cam_max"] = len(rank_lst)
        if rank_lst2 is not None:
            save_path = base_path + "/ps_data/{}_{}_rank_list.npy".format(name, "ctm")
            np.save(save_path, rank_lst2)
            csv_data["ctm_t_selection"] = t_selection_ctm
        if name != "DeepGini":
            csv_data["rate"] = rate
        df = add_df(df, csv_data)
    return df


# 获得返回值的map
def get_ps_csv_data():
    ps_collection_data = {
        "name": None,
        "rate": None,
        "t_collection": None,
        "cam_t_selection": None,
        "cam_max": None,
        "ctm_t_selection": None,
    }
    return ps_collection_data


# 向dataFrame里添加一条数据
def add_df(df, csv_data):
    if df is None:  # 如果是空的
        df = pd.DataFrame(csv_data, index=[0])
    else:
        df.loc[df.shape[0]] = csv_data
    return df


def get_div_data(Xa_train, Ya_train, base_path, select_size, b, ub, select_pnum=None, is_anlyze=False,
                 is_get_profile=False):
    ori_model = load_model(model_conf.get_exp2_model_path(params["data_name"], params["model_name"]))
    csv_data = get_ps_csv_data()
    # 计算div
    # 设定参数&计算数据
    cov_config.boundary = b
    cov_config.up_boundary = ub
    if select_pnum is not None:
        cov_config.select_pnum = select_pnum
    # div 所有的数据结果
    if is_get_profile:
        deepdiv.cal_d_v(Xa_train, Ya_train, params["nb_classes"], ori_model, base_path=base_path,
                        is_anlyze=is_get_profile)  # CK_point
    #  选出3000个
    target_size = int(select_size / params["nb_classes"])
    extra_size = int(select_size % params["nb_classes"])
    s = time.time()
    Xr_select, select_lable_arr, c_arr, max_size_arr = deepdiv.datasets_select(Xa_train, Ya_train, 10, ori_model,
                                                                               target_size,
                                                                               extra_size=extra_size,
                                                                               base_path=base_path,
                                                                               is_analyze=is_anlyze)  # cov_change
    e = time.time()
    Xr_idx = np.concatenate(Xr_select, axis=0)
    save_path = base_path + "/ps_data/{}_{}_rank_list.npy".format("Div_select", select_size)
    np.save(save_path, Xr_idx)
    csv_data["name"] = "Div_select"
    csv_data["rate"] = np.round(np.mean(c_arr), 2)
    csv_data["cam_t_selection"] = e - s
    # csv_data["t_collection"] = t_collection
    return csv_data


#############################################
# 重训练
#############################################

# retrain后的实验结果模板
def get_retrain_csv_data(name, method, acc_imp, acc_si_train, t_retrain, size):
    csv_data = {}
    csv_data["name"] = name
    csv_data["method"] = method
    csv_data["acc_imp_ori"] = acc_imp
    csv_data["acc_imp_val"] = acc_si_train
    csv_data["t_retrain"] = t_retrain
    csv_data["size"] = size
    return csv_data


# retrain model
def get_retrain_res(x_s, y_s, verbose=True):
    K.clear_session()
    X_train, X_test, Y_train, Y_test, _, _, X_val, Y_val = prepare_data()
    # 1 . 合并训练集
    Ya_train = np.concatenate([Y_train, y_s])
    Xa_train = np.concatenate([X_train, x_s])
    # 2. hot
    Ya_train_vec = keras.utils.np_utils.to_categorical(Ya_train, params["nb_classes"])
    Y_test_vec = keras.utils.np_utils.to_categorical(Y_test, params["nb_classes"])
    Y_val_vec = keras.utils.np_utils.to_categorical(Y_val, params["nb_classes"])
    # 2.
    ori_model = load_model(model_conf.get_exp2_model_path(params["data_name"], params["model_name"]))
    # 在 测试集上的精度  准确性
    acc_base_ori = ori_model.evaluate(X_test, Y_test_vec, verbose=0)[1]
    # 在 验证集上的精度  泛化鲁邦性
    acc_base_val = ori_model.evaluate(X_val, Y_val_vec, verbose=0)[1]
    sss = time.time()
    trained_model = retrain_model(params, ori_model, Xa_train, Ya_train_vec, X_val, Y_val_vec, "cov", 0,
                                  verbose=verbose)
    eee = time.time()
    acc_si_ori = trained_model.evaluate(X_test, Y_test_vec, verbose=0)[1]
    acc_si_val = trained_model.evaluate(X_val, Y_val_vec, verbose=0)[1]

    acc_imp_ori = acc_si_ori - acc_base_ori
    acc_imp_val = acc_si_val - acc_base_val

    print("ori acc", acc_base_ori, acc_si_ori, "diff:", format(acc_imp_ori, ".3f"))
    print("val acc", acc_base_val, acc_si_val, "diff:", format(acc_imp_val, ".3f"))
    K.clear_session()  # 每次重训练后都清缓存
    return acc_imp_ori, acc_imp_val, eee - sss


# 根据下标去获得数据
def get_retrain_data_by_idx(Xa_train, Ya_train, idx, is_val_data=False):
    x_s, y_s = Xa_train[idx], Ya_train[idx]
    return x_s, y_s


# 1.实现cov idx当cam不够时,使用随机补
def get_retrain_idx(cam_path, ctm_path, select_size, Xa_train_size):
    print(cam_path)
    print(ctm_path)
    ctm_ps_arr, cam_ps_arr = None, None
    ctm_idx, cam_idx = None, None
    if os.path.exists(ctm_path):
        print("ctm exits")
        ctm_ps_arr = np.load(ctm_path)
        ctm_idx = ctm_ps_arr[:select_size]
    if os.path.exists(cam_path):
        print("cam exits")
        cam_ps_arr = np.load(cam_path)
        if len(cam_ps_arr) >= select_size:
            cam_idx = cam_ps_arr[:select_size]
        else:  # cam不够,随机机选补充
            # if ctm_ps_arr is None:
            print("cam is max ,use random ")
            diff_size = select_size - len(cam_ps_arr)  # cam全放里不够,剩下的随机补
            diff_idx = list(set(list(range(Xa_train_size))) - set(cam_ps_arr))  # 剩下的测试用例
            random.seed(42)
            idx = random.sample(diff_idx, diff_size)  # 随机选差的数量
            cam_idx = list(cam_ps_arr) + list(idx)  # 将cam的序列放前面,随机的放后面,
    if cam_idx is not None and len(cam_idx) != select_size:
        raise ValueError("cam选出的用例数与预期不一致!")

    if ctm_idx is not None and len(ctm_idx) != select_size:
        raise ValueError("ctm选出的用例数与预期不一致!")
    return cam_idx, ctm_idx


# 2.获得cov指标的重训练下标顺序,
def get_cov_retrain_idx(name, select_size, Xa_train_len, ps_path, ):  #
    idx_data = {}
    ctm_path = ps_path + "{}_{}_rank_list.npy".format(name, "ctm")
    cam_path = ps_path + "{}_{}_rank_list.npy".format(name, "cam")
    cam_idx, ctm_idx = get_retrain_idx(cam_path, ctm_path, select_size, Xa_train_len)
    idx_arr = [cam_idx, ctm_idx]
    prefix_arr = ['cam', "ctm"]
    for prefix, idx in zip(prefix_arr, idx_arr):
        if idx is None:
            continue
        idx_data[name + "_" + prefix] = idx
    return idx_data


# 3.获得div指标下的重训练下标顺序
def get_div_retrain_idx(ps_path, select_size, ):
    path = ps_path + "{}_{}_rank_list.npy".format("Div_select", select_size)
    idx = np.load(path)
    return idx


# 4. 获得random的下标
def get_random_retrain_idx(select_size, Xa_train_len, seed=None, ):
    if seed is not None:
        np.random.seed(seed)
    shuffle_indices = np.random.permutation(Xa_train_len)
    shuffle_indices_select = shuffle_indices[:select_size]
    return shuffle_indices_select


# 5.random比较复杂 ,单独抽出一个方法来写
def get_random_retrain_data(df, select_size, verbose=True):
    _, _, _, _, X_select, Y_select, _, _ = prepare_data()
    name = "Random"
    prefix = "ALL"
    idx = get_random_retrain_idx(select_size, len(X_select), seed=None, )
    x_s, y_s = get_retrain_data_by_idx(X_select, Y_select, idx, )  # 增加数据分析
    acc_imp_ori, acc_imp_val, retrain_time = get_retrain_res(x_s, y_s,
                                                             verbose=verbose)  # 增加训练后模型与数据分析
    trained_csv_data = get_retrain_csv_data(name, prefix, acc_imp_ori, acc_imp_val, retrain_time, len(x_s))
    df = add_df(df, trained_csv_data)  # 0.9515 0.9738

    return df, idx


# 本次重训练的数据大小,使用指标,和Ps路径  数据的规模大小
############
############
def exp_retrain(base_path, select_size, cov_name_list=None, verbose=True, is_retrain_cov=True, is_retrain_div=True,
                is_retrain_random=True):
    ps_path = "{}/ps_data/".format(base_path)
    csv_path = base_path + "/test_exp_{}_retrain.csv".format(select_size)
    idx_path = base_path + "/test_exp_{}_idx.csv".format(select_size)
    _, X_test, _, _, X_select, Y_select, _, _ = prepare_data()

    res_idx_map = {}
    df = None
    if is_retrain_cov:
        # 1. cov
        for name in cov_name_list:
            temp_idx_data = get_cov_retrain_idx(name, select_size, len(X_select), ps_path, )
            # 1.合并选取的下标, 用于实验3
            res_idx_map = dict(res_idx_map, **temp_idx_data)
            # 2. 选取数据,重训练
            for k, idx in temp_idx_data.items():
                prefix = str(k).split("_")[-1]
                x_s, y_s = get_retrain_data_by_idx(X_select, Y_select, idx, )
                acc_imp_ori, acc_imp_val, retrain_time = get_retrain_res(x_s, y_s,
                                                                         verbose=verbose)
                cov_trained_csv_data = get_retrain_csv_data(name, prefix, acc_imp_ori, acc_imp_val, retrain_time,
                                                            len(x_s))
                df = add_df(df, cov_trained_csv_data)

    # 2.div
    if is_retrain_div:
        div_name = "Div_select"
        div_idx = get_div_retrain_idx(ps_path, select_size, )
        res_idx_map[div_name] = div_idx  # 用于实验3
        x_s, y_s = get_retrain_data_by_idx(X_select, Y_select, div_idx, )
        acc_imp_ori, acc_imp_val, retrain_time = get_retrain_res(x_s, y_s,
                                                                 verbose=verbose)
        cov_trained_csv_data = get_retrain_csv_data(div_name, "None", acc_imp_ori, acc_imp_val, retrain_time,
                                                    len(x_s))
        df = add_df(df, cov_trained_csv_data)
    if is_retrain_random:
        # 3.random
        df, idx = get_random_retrain_data(df, select_size, verbose=verbose)
        res_idx_map["Random"] = idx  # 用于实验3
    try:
        # df["dau_per"] = 0  # 新增一列
        for k, v in res_idx_map.items():
            select_dau_per = np.sum((np.array(v) >= len(X_test))) / (len(X_select) - len(X_test))  # 选取的占总共的
            select_dau_per2 = np.sum((np.array(v) >= len(X_test))) / (len(v))  # 选取的占当前的
            # df.loc[k, "dau_per"] = format(select_dau_per, ".2f")
            print(select_dau_per, select_dau_per2)
    except:
        print("error")
    df.to_csv(csv_path, index=False)
    df_idx = pd.DataFrame(res_idx_map)
    df_idx.to_csv(idx_path, index=False)


def exp(params):
    # 实验参数
    exp_params = {
        "k_list": [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],  # 数据size总大小
        "b_list": [0],
        "ub_list": [0.99],
        "select_pnum": 2,  # not use
        "step_th": 0.005,  # not use
        "exp_name": "rq2  select effect",  # 实验名称
        "modify": "None"
    }

    # 用于优先级序列
    b_list, ub_list = exp_params["b_list"], exp_params["ub_list"]  # 不分块  只选取高置信度部分,低的不要 # > 0.22
    b = b_list[0]
    ub = ub_list[0]
    cov_config.step_th = exp_params["step_th"]
    sp = exp_params["select_pnum"]
    is_get_profile = False  # 获得所有数据的最大div
    # 用于控制实验流程
    is_prepare_ps = True  # 是否准备ps
    is_prepare_cov_ps = True
    is_retrain = True  # 是否重训练
    is_retrain_div = False  # 是否重训练div
    is_retrain_cov = True  # 两个false就只重训练div
    is_retrain_random = False
    base_path = None  # 如果不准备ps,直接retrain,就填写ps的路径,
    verbose = 2  # 打印模型训练信息
    # 1 获得原始数据
    X_train, _, Y_train, _, X_select, _, _, _ = prepare_data()

    # 2.获得伪标签
    ori_model = load_model(model_conf.get_exp2_model_path(params["data_name"], params["model_name"]))
    prob_matrixc = ori_model.predict(X_select)
    Y_psedu_select = np.argmax(prob_matrixc, axis=1)  # 每行最大的置信度
    # 构建指标(准备ps时,都使用伪标签)
    max_select_size = exp_params['k_list'][-1] * len(X_select)  # 最大选择数量
    cover = neural_cov(params["model_name"], params["data_name"], X_select, Y_psedu_select, X_train, Y_train)
    cov_name_list, func_list = get_cov_name_and_func(cover, max_select_size, use_max=False)
    if is_prepare_ps:
        # 5. 创建实验文件夹
        df = None
        base_path = mk_exp_dir(dict(params, **exp_params))
        ps_csv_path = base_path + "/exp_ps_collection.csv"
        # 6 获取实验指标
        if is_prepare_cov_ps:
            # 准备cov 的ps
            print("ppppp")
            df = get_cov_data(base_path, cov_name_list, func_list, None, )
        # 准备div 的ps
        for k in exp_params['k_list']:
            select_size = int(len(X_select) * k)
            csv_data = get_div_data(X_select, Y_psedu_select, base_path, select_size, b, ub, select_pnum=sp,
                                    is_anlyze=True, is_get_profile=is_get_profile)
            df = add_df(df, csv_data)
            df.to_csv(ps_csv_path, index=False)
            if is_retrain:
                # 重训练
                exp_retrain(base_path, select_size, cov_name_list=cov_name_list,
                            verbose=verbose, is_retrain_cov=is_retrain_cov,
                            is_retrain_random=is_retrain_random, is_retrain_div=is_retrain_div)
    statistic_data(base_path, exp_params["k_list"], len(X_select))


################################
# 绘图
###
# 打印retrain表格
def statistic_data(base_path, k_list, len_X_select):
    table_path = base_path + "/table"
    fig_path = base_path + "/fig"
    imp_val_csv_path = table_path + "/{}_{}_{}.csv".format(params["data_name"], params["model_name"], "imp_val")
    imp_ori_csv_path = table_path + "/{}_{}_{}.csv".format(params["data_name"], params["model_name"], "imp_ori")
    imp_val_fig_path = fig_path + "/{}_{}_{}.png".format(params["data_name"], params["model_name"], "imp_val")
    imp_ori_fig_path = fig_path + "/{}_{}_{}.png".format(params["data_name"], params["model_name"], "imp_ori")
    os.makedirs(table_path, exist_ok=True)
    os.makedirs(fig_path, exist_ok=True)

    def table():
        # 表格打印
        df_imp_val = pd.DataFrame()
        df_imp_ori = pd.DataFrame()
        k = k_list[0]
        select_size = int(len_X_select * k)
        temp_path = base_path + "/test_exp_{}_retrain.csv".format(select_size)
        temp_df = pd.read_csv(temp_path)

        for key in ["name", "method"]:
            arr = temp_df[key]
            df_imp_val[key] = arr
            df_imp_ori[key] = arr

        for k in k_list:
            select_size = int(len_X_select * k)
            csv_path = base_path + "/test_exp_{}_retrain.csv".format(select_size)
            df = pd.read_csv(csv_path)

            acc_imp_val = df["acc_imp_val"]
            df_imp_val[str(k * 100)] = acc_imp_val

            acc_imp_ori = df["acc_imp_ori"]
            df_imp_ori[str(k * 100)] = acc_imp_ori

        df_imp_val.to_csv(imp_val_csv_path)
        df_imp_ori.to_csv(imp_ori_csv_path)

    table()

    def fig():
        import matplotlib.pyplot as plt
        df_imp_val = pd.read_csv(imp_val_csv_path)
        df_imp_ori = pd.read_csv(imp_ori_csv_path)
        # 画图
        x_sticks = range(len(k_list))
        x_ticklabels = [str(int(k * 100)) for k in k_list]
        x_ticklabels.insert(0, 0)
        df_arr = [df_imp_val, df_imp_ori]
        path_arr = [imp_val_fig_path, imp_ori_fig_path]
        title_arr = ["imp_val", "imp_ori"]
        for df, p, title in zip(df_arr, path_arr, title_arr):
            for index, row in df.iterrows():
                label = row["name"] + "_" + row["method"]
                data = row[3:]
                if "Div_select" in label:
                    plt.plot(x_sticks, data, label=label, color="crimson")
                elif "DeepGini" in label:
                    # plt.plot(x_sticks, data, label=label, color="darkgreen")
                    continue
                elif "Random" in label:
                    plt.plot(x_sticks, data, label=label, color="black")
                else:
                    plt.plot(x_sticks, data, label=label, alpha=0.5)
            ax = plt.gca()
            ax.set_xticklabels(x_ticklabels)
            plt.xlabel("top th ")
            plt.ylabel("acc_imp")
            plt.title(title)
            plt.legend()
            plt.savefig(p)
            plt.close()

    fig()


train_dau_num = 1
test_dau_num = 1
exp_name = "exp2_kmnc"
params = None  # 待赋值


def set_params(p):
    global params
    params = p


def run(p, gpu_no=0, ename=None, train_dn=1, test_dn=1):
    global params
    global train_dau_num
    global test_dau_num
    params = p
    train_dau_num = train_dn
    test_dau_num = test_dn
    if ename is not None:
        global exp_name
        exp_name = ename
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_no)
    prepare_model()
    exp(params)


if __name__ == '__main__':
    params = {
        "model_name": model_conf.LeNet1,  # 选取模型
        "data_name": model_conf.mnist,  # 选取数据集
        "nb_classes": model_conf.fig_nb_classes,  # 数据集分类数
        "dau_name": "mnist_harder"  # 选取扩增数据集名称
    }
    run(params, 0)
