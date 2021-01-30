# 实验4 时间复杂度,补充div的 cam数量
import functools
import os
import time

from keras.engine.saving import load_model

import model_conf
import deepdiv
from gen_model.gen_model_utils import gen_exp2_model
from neural_cov import neural_cov
from exp2 import prepare_data, set_params, get_cov_data, get_cov_name_and_func
import pandas as pd
import numpy as np
from keras import backend as K


def get_exp2_data(v, k):
    params = {
        "model_name": v,
        "data_name": k,
        "nb_classes": model_conf.fig_nb_classes,
        "dau_name": k + "_harder"
    }
    set_params(p=params)
    # 1 获得原始数据
    X_train, X_test, Y_train, Y_test, X_select, _, _, _ = prepare_data()
    return X_train, Y_train, X_select, X_test, Y_test


# 补充每个数据集+模型的div的最大覆盖
def exp4_div(model_data):
    res_path = "result/exp4"
    is_save_ps = True
    use_add = False
    time_cost_path = os.path.join(res_path, "time_cost.csv")
    csv_data = {}
    for k, v_arr in model_data.items():
        for v in v_arr:
            pair_name = model_conf.get_pair_name(k, v)
            base_path = os.path.join(res_path, pair_name)
            os.makedirs(base_path, exist_ok=True)
            X_train, Y_train, X_select, X_test, Y_test = get_exp2_data(v, k)
            # 2.获得伪标签
            ori_model = load_model(model_conf.get_exp2_model_path(k, v))
            s = time.time()
            prob_matrixc = ori_model.predict(X_select)
            Y_psedu_select = np.argmax(prob_matrixc, axis=1)  # 每行最大的置信度
            e = time.time()
            csv_data[pair_name + "_collect"] = e - s

            s = time.time()
            deepdiv.get_priority_sequence(X_select, Y_psedu_select, model_conf.fig_nb_classes, ori_model,
                                          base_path=base_path,
                                          use_add=use_add, is_save_ps=is_save_ps)
            e = time.time()
            csv_data[pair_name + "_select"] = e - s
            K.clear_session()
    df = pd.DataFrame(csv_data, index=[0])
    df.to_csv(time_cost_path, index=False)


# if you only want to see neural cov's time_cost
def exp4_cov(model_data):
    res_path = "result/exp4/cov"
    for k, v_arr in model_data.items():
        for v in v_arr:
            df = None
            pair_name = model_conf.get_pair_name(k, v)
            base_path = os.path.join(res_path, pair_name)  # base_path  # 每个模型_数据集的位置
            time_cost_path = os.path.join(base_path, "cov_time_cost.csv")  # 文件名
            os.makedirs(base_path + "/ps_data", exist_ok=True)
            # 准备数据
            X_train, Y_train, X_select, X_test, Y_test = get_exp2_data(v, k)
            gen_exp2_model(k, v, X_train, Y_train, X_test, Y_test, )
            # 2.获得伪标签
            ori_model = load_model(model_conf.get_exp2_model_path(k, v))
            prob_matrixc = ori_model.predict(X_select)
            Y_psedu_select = np.argmax(prob_matrixc, axis=1)  # 每行最大的置信度
            # 准备覆盖
            cover = neural_cov(v, k, X_select, Y_psedu_select, X_train, Y_train)
            cov_name_list, func_list = get_cov_name_and_func(cover)

            del cov_name_list[0]
            del func_list[0]

            cov_name_list.append("DSC")
            cal_dsc_cov_limit = functools.partial(cover.cal_dsc_cov, time_limit=36000)
            func_list.append(cal_dsc_cov_limit)
            if v == model_conf.LeNet1:
                cov_name_list.append("KMNC")
                cal_kmnc_cov_limit = functools.partial(cover.cal_kmnc_cov, max_select_size=None, time_limit=36000)
                func_list.append(cal_kmnc_cov_limit)
            # 进行试验
            df = get_cov_data(base_path, cov_name_list, func_list, df, )

            df.to_csv(time_cost_path, index=False)
            K.clear_session()


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    model_data = model_conf.model_data
    exp4_div(model_data)
