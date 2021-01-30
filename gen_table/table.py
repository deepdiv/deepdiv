import abc
import collections

import pandas as pd
import numpy as np
import glob
import os
import model_conf
from pandas import MultiIndex

from exp1 import get_exp1_test_size, get_exp1_fault_size
from exp2 import get_len_data


# table cols and rows name
# exp's params
# There is no need to modify this code
class MyTable(object):

    def __init__(self, table_name):
        self.table_path = "table"
        self.table_name = table_name
        self.exp2_k_list = ["1.0", "5.0", "10.0", "15.0", "20.0", "25.0", "30.0", ]  # 两者综合
        self.exp2_k_list2 = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]  # 实验2的
        self.exp1_k_list = [0.1, 0.3]
        os.makedirs(self.table_path, exist_ok=True)

    # 表格1的数据长度
    @staticmethod
    def get_exp1_size(data_name, dau_name):
        len_X = get_exp1_test_size(data_name, dau_name)
        return len_X

    # 表格1的错误数据大小,计算错误比
    @staticmethod
    def get_exp1_fault_size(data_name, dau_name, model_name):
        len_X = get_exp1_fault_size(data_name, dau_name, model_name)
        return len_X

    # 表格2的数据长度
    @staticmethod
    def get_exp2_size(data_name, dau_name):
        len_X_select = get_len_data(data_name, dau_name)
        return len_X_select

    # 表格2,3,4 的横轴数据集顺序(读取)
    @staticmethod
    def get_order_model_data():
        sortorder = {model_conf.mnist: 0, model_conf.fashion: 1, model_conf.cifar10: 2, model_conf.svhn: 3, }
        model_data = model_conf.model_data
        # print(model_data)
        model_data = sorted(model_data.items(), key=lambda x: sortorder[x[0]])
        # print(model_data)
        d1 = collections.OrderedDict()
        for (k, v) in model_data:
            d1[k] = v
        return d1

    @staticmethod
    def get_order_pair_name():
        res = []
        model_data = MyTable.get_order_model_data()
        for (k, v_arr) in model_data.items():
            for v in v_arr:
                res.append("{}_{}".format(k, v))
        return res

    def add_df(self, df, csv_data):
        if df is None:  # 如果是空的
            df = pd.DataFrame(csv_data, index=[0])
        else:
            df.loc[df.shape[0]] = csv_data
        return df

    # 初始化表格
    @abc.abstractmethod
    def init_table(self):
        pass

    @staticmethod
    def num_to_str(num, trunc=2):
        return format(num, '.{}f'.format(trunc))

    # @staticmethod
    # def get_metrics_name():
    #     data_name_list = ["Random", "NAC", "NBC", "SNAC", "KMNC", "TKNC", "LSC", "DSC", "DeepGini", "Div_select", ]
    #     return data_name_list,

    # 获得方法名对应的排序方法
    @staticmethod
    def get_metrics_method():
        params = {
            "Random": ["ALL"],
            "NAC": ["cam", "ctm"],
            "NBC": ["cam", "ctm"],
            "SNAC": ["cam", "ctm"],
            "TKNC": ["cam", ],
            "KMNC": ["cam", ],
            "LSC": ["cam", ],
            "DSC": ["cam", ],
            "DeepGini": ["ctm", ],
            "Div_select": ["None", ],
        }
        return params

    # 根据原始数据表格获得table1的列名
    @staticmethod
    def get_table_cols_by_name(name):
        params = {
            "cov_nac": "NAC",
            "cov_nbc": "NBC",
            "cov_snac": "SNAC",
            "cov_tknc": "TKNC",
            "cov_kmnc": "KMNC",
            "cov_lsc": "LSC",
            "cov_dsc": "DSC",
            "div": "Div",
        }
        return params[name]

    # 获得table23,的行名
    @staticmethod
    def get_table23_cols():
        # table_name_list = ["Random", "NAC(0.75)-CAM", "NAC(0.75)-CTM", "NBC(0)-CAM", "NBC(0)-CTM", "SNAC(0)-CAM",
        #                    "SNAC(0)-CTM", "KMNC(1000)-CAM", "TKNC(1)-CAM", "LSC(1000,100)-CAM", "DSC(1000,100)-CAM",
        #                    "DeepGini",
        #                    "Div_select"]
        table_name_list = ["Random", "NAC(0.75)-CAM", "NAC(0.75)-CTM", "NBC(0)-CAM", "NBC(0)-CTM", "SNAC(0)-CAM",
                           "SNAC(0)-CTM", "KMNC(1000)-CAM", "TKNC(1)-CAM", "LSC(1000,100)-CAM", "DSC(1000,100)-CAM",
                           "Div_select"]
        return table_name_list

    # 获得table4的行名
    @staticmethod
    def get_table4_cols():
        # table_name_list = ["Random", "NAC(0.75)-CAM", "NAC(0.75)-CTM", "NBC(0)-CAM", "NBC(0)-CTM", "SNAC(0)-CAM",
        #                    "SNAC(0)-CTM", "KMNC(1000)-CAM", "TKNC(1)-CAM", "LSC(1000,100)-CAM", "DSC(1000,100)-CAM",
        #                    "DeepGini",
        #                    "Div_select"]
        table_name_list = ["NAC(0.75)", "NBC(0)", "SNAC(0)", "TKNC(1)", "KMNC(1000)", "LSC(1000,100)", "DSC(1000,100)",
                           "DeepDiv"]
        return table_name_list

    # 根据原始数据表格获得table23的行名
    @staticmethod
    def get_table23_cols_by_name_and_method(name, method):
        if name == "Random":
            return "Random"
        if name == "NAC":
            return "NAC(0.75)-{}".format(str(method).upper())
        if name == "NBC":
            return "NBC(0)-{}".format(str(method).upper())
        if name == "SNAC":
            return "SNAC(0)-{}".format(str(method).upper())
        if name == "KMNC":
            return "KMNC(1000)-{}".format(str(method).upper())
        if name == "TKNC":
            return "TKNC(1)-{}".format(str(method).upper())
        if name == "LSC":
            return "LSC(1000,100)-{}".format(str(method).upper())
        if name == "DSC":
            return "DSC(1000,100)-{}".format(str(method).upper())
        if name == "DeepGini":
            return "DeepGini"
        if name == "Div_select":
            return "Div_select"
        raise Exception("no methods")

    # 根据原始数据表格获得table4的行名
    @staticmethod
    def get_table4_cols_by_name(name):
        if name == "Random":
            return "Random"
        if name == "NAC":
            return "NAC(0.75)"
        if name == "NBC":
            return "NBC(0)"
        if name == "SNAC":
            return "SNAC(0)"
        if name == "KMNC":
            return "KMNC(1000)"
        if name == "TKNC":
            return "TKNC(1)"
        if name == "LSC":
            return "LSC(1000,100)"
        if name == "DSC":
            return "DSC(1000,100)"
        if name == "DeepGini":
            return "DeepGini"
        if name == "Div_select":
            return "DeepDiv"
        raise Exception("no methods")
