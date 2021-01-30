import time

import numpy as np
from keras.engine.saving import load_model

import metrics
##############
# 计算神经元覆盖率, 时间, 排序后结果
#############
import model_conf


# 用于实验2
# train 和select都是固定的
# 看select的最大覆盖和排序
class neural_cov(object):
    def __init__(self, model_name, data_name, x_s, y_s, X_train, Y_train, ):
        self.Y_train = Y_train
        self.X_train = X_train
        self.y_s = y_s
        self.x_s = x_s
        self.ori_model = load_model(model_conf.get_exp2_model_path(data_name, model_name))
        self.model_name = model_name
        self.data_name = data_name

    def get_layers(self):
        input_layer, layers = metrics.get_layers(self.model_name, self.ori_model)
        return input_layer, layers

    def cal_deepgini(self):
        s = time.time()
        pred_test_prob = self.ori_model.predict(self.x_s)
        e = time.time()
        t_collection = e - s

        s = time.time()
        metrics = np.sum(pred_test_prob ** 2, axis=1)  # 值越小,1-值就越大,因此值越小越好
        rank_lst = np.argsort(metrics)  # 按照值从小到大排序,因此序号越小代表值越小代表越好
        e = time.time()
        t_selection_cam = e - s
        return 1 - metrics, t_collection, None, None, rank_lst, t_selection_cam,

    def cal_nac_cov(self, t=0.0):
        input_layer, layers = metrics.get_layers(self.model_name, self.ori_model)
        # 获取度量指标
        nac = metrics.nac(self.x_s, input_layer, layers, t=t)
        # 覆盖率计算
        rate = nac.fit()
        # 排序
        rank_lst = nac.rank_fast(self.x_s, )
        # 排序2
        rank_lst2 = nac.rank_2(self.x_s, )
        return rate, nac.t_collection, rank_lst, nac.t_cam, rank_lst2, nac.t_ctm

    # 可以贪心排,可以单个样本比较排
    def cal_nbc_cov(self, std=0):  # 0 0.5 1
        input_layer, layers = metrics.get_layers(self.model_name, self.ori_model)
        nbc = metrics.nbc(self.X_train, input_layer, layers, std=std)

        rate = nbc.fit(self.x_s, use_lower=True)

        rank_lst = nbc.rank_fast(self.x_s, use_lower=True)

        rank_lst2 = nbc.rank_2(self.x_s, use_lower=True)

        return rate, nbc.t_collection, rank_lst, nbc.t_cam, rank_lst2, nbc.t_ctm

    # 可以贪心排,可以单个样本比较排
    def cal_snac_cov(self, std=0):  # 0 0.5 1
        input_layer, layers = metrics.get_layers(self.model_name, self.ori_model)
        snac = metrics.nbc(self.X_train, input_layer, layers, std=std)

        rate = snac.fit(self.x_s, use_lower=False)

        rank_lst = snac.rank_fast(self.x_s, use_lower=False)

        rank_lst2 = snac.rank_2(self.x_s, use_lower=False)

        return rate, snac.t_collection, rank_lst, snac.t_cam, rank_lst2, snac.t_ctm

    # 只能贪心排
    def cal_kmnc_cov(self, k_bins=1000, max_select_size=None, time_limit=3600):  # 1000
        input_layer, layers = metrics.get_layers(self.model_name, self.ori_model)
        kmnc = metrics.kmnc(self.X_train, input_layer, layers, k_bins=k_bins, max_select_size=max_select_size,
                            time_limit=time_limit)

        rate = kmnc.fit(self.x_s, )

        rank_lst = kmnc.rank_fast(self.x_s, )

        rank_lst2 = None

        return rate, kmnc.t_collection, rank_lst, kmnc.t_cam, rank_lst2, None

    # 只能贪心排
    def cal_tknc_cov(self, k=2):  # 1,2,3
        input_layer, layers = metrics.get_layers(self.model_name, self.ori_model)
        tknc = metrics.tknc(self.x_s, input_layer, layers, k=k)

        rate = tknc.fit(list(range(len(self.x_s))))

        rank_lst = tknc.rank(self.x_s, )

        rank_lst2 = None

        return rate, tknc.t_collection, rank_lst, tknc.t_cam, rank_lst2, None

    # 单个用例的ctm的覆盖率相同
    def cal_lsc_cov(self, k_bins=1000, u=100, index=-1):
        input_layer, layers = metrics.get_layers(self.model_name, self.ori_model)
        lsc = metrics.LSC(self.X_train, self.Y_train, input_layer, [layers[index]], k_bins=k_bins, u=u)

        # lsc dsc中的排序使用了fit中的中间结果,因此时间成本要加入进去
        rate = lsc.fit(self.x_s, self.y_s)

        rank_lst = lsc.rank_fast()

        # return rate, t_collection, rank_lst, t_selection_cam, rank_lst2, t_selection_ctm
        return rate, lsc.t_collection, rank_lst, lsc.t_cam, None, None

    def cal_dsc_cov(self, k_bins=1000, u=2, index=-1, time_limit=3600):
        input_layer, layers = metrics.get_layers(self.model_name, self.ori_model)
        dsc = metrics.DSC(self.X_train, self.Y_train, input_layer, [layers[index]], k_bins=k_bins, u=u,
                          time_limit=time_limit)

        rate = dsc.fit(self.x_s, self.y_s)

        rank_lst = dsc.rank_fast()

        return rate, dsc.t_collection, rank_lst, dsc.t_cam, None, None


# 用于实验1
# train是固定的,test是变化的
# 看test的覆盖大小
# train不变的情况下,减少t_collection的时间
class CovInit(object):
    def __init__(self, X_train, Y_train, params):
        ori_model = load_model(model_conf.get_model_path(params["data_name"], params["model_name"]))
        input_layer, layers = metrics.get_layers(params["model_name"], ori_model)
        # 当给定数据集与模型的时候,输入和测试层是固定的
        self.input_layer = input_layer
        self.layers = layers
        # 当给定数据集的时候,训练集是固定的
        self.X_train = X_train
        self.Y_train = Y_train

    def get_input_layer(self):
        return self.input_layer

    def get_layers(self):
        return self.layers

    def get_nbc(self):
        nbc = metrics.nbc(self.X_train, self.input_layer, self.layers)
        return nbc

    def get_kmnc(self):
        kmnc = metrics.kmnc(self.X_train, self.input_layer, self.layers)
        return kmnc

    def get_lsc(self, k_bins=1000, index=-1):
        lsc = metrics.LSC(self.X_train, self.Y_train, self.input_layer, [self.layers[index]], k_bins=k_bins, u=100)
        return lsc

    def get_dsc(self, k_bins=1000):
        dsc = metrics.DSC(self.X_train, self.Y_train, self.input_layer, self.layers, k_bins=k_bins)
        return dsc
