import os

import matplotlib

import model_conf
from gen_table.table import MyTable
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import seaborn
import matplotlib.pyplot as plt
import numpy as np


# path template
# model_data = {
#     mnist: [LeNet5, LeNet1],
#     fashion: [LeNet1, resNet20],
#     cifar10: [vgg16, resNet20],
#     svhn: [LeNet5, vgg16]
# }

# first we merge the exp result to merge_res/
# all table or figs base on the merge_res/
# We just need to modify the res path ,then we can get tables and figs
# 将实验结果整合到一起
class ResMerger(object):
    # 表格1
    # you should write exp1 result file_path here as path template
    @staticmethod
    def get_exp1_suffix_dict():
        suffix_dict = {
            model_conf.mnist: ["01091740_exp1", "01081738_exp1"],
            model_conf.fashion: ["01081741_exp1", "01081908_exp1"],
            model_conf.cifar10: ["01100656_exp1", "01091611_exp1"],
            model_conf.svhn: ["01081905_exp1", "01082236_exp1"]
        }
        return suffix_dict

    # 表格1 dsc
    # you should write exp1_dsc result file_path here as path template
    @staticmethod
    def get_exp1_dsc_suffix_dict():
        suffix_dict = {
            model_conf.mnist: ["01101837_exp1_dsc", "01101814_exp1_dsc"],
            model_conf.fashion: ["01101951_exp1_dsc",
                                 "01102014_exp1_dsc"],
            model_conf.cifar10: ["01151801_exp1_dsc", "01151341_exp1_dsc"],
            model_conf.svhn: ["01110043_exp1_dsc", None],
        }
        return suffix_dict

    # 获取 table234所在的实验文件夹
    @staticmethod
    # you should write exp2 result file_path here as path template
    def get_exp2_suffix_dict():
        suffix_dict = {
            model_conf.mnist: ["12200913_exp2", "12201718_exp2"],
            model_conf.fashion: ["12201022_exp2", "12201820_exp2"],
            model_conf.cifar10: ["12201130_exp2", "12210039_exp2"],
            model_conf.svhn: ["12201458_exp2", "12210608_exp2"]
        }
        return suffix_dict

    # 234 dsc
    @staticmethod
    # you should write exp2_dsc result file_path here as path template
    def get_exp2_dsc_suffix_dict():
        suffix_dict = {
            model_conf.mnist: ["01040649_exp2_dsc", "01040611_exp2_dsc"],
            model_conf.fashion: ["01040634_exp2_dsc", "01040739_exp2_dsc"],
            model_conf.cifar10: ["01041159_exp2_dsc", "01041222_exp2_dsc"],
            model_conf.svhn: ["01040700_exp2_dsc", "01041118_exp2_dsc", ]
        }
        return suffix_dict

    # 234 kmnc
    @staticmethod
    # you should write exp2_kmnc result file_path here as path template
    def get_exp2_kmnc_suffix_dict():
        suffix_dict = {
            model_conf.mnist: [None, "01040424_exp2_kmnc"],
            model_conf.fashion: ["12201611_exp2_kmnc", None],
            model_conf.cifar10: [None, None],
            model_conf.svhn: [None, None]
        }
        return suffix_dict

    # 合并实验2 (RQ234的文件)
    # merge exp2 result. merge_res/exp2
    def merge_exp2(self):
        def get_dir_path(su):
            base_path = "result"
            dir_name = su + "_" + model_conf.get_pair_name(key, v)
            dir_path = "{}/{}".format(base_path, dir_name)
            return dir_path

        def get_to_path(to_base_path, pair_name, fs):
            return os.path.join(to_base_path, pair_name, fs)

        # 三个目标文件
        f_retrain = "test_exp_{}_retrain.csv"  # 重训练精度 RQ2
        f_idx = "test_exp_{}_idx.csv"  # 相似度 RQ3
        f_ps = "exp_ps_collection.csv"  # 时间复杂度 RQ4
        # 合并
        to_base_path = "merge_res/exp2"
        model_data = MyTable.get_order_model_data()
        suffix_dict = ResMerger.get_exp2_suffix_dict()
        kmnc_suffix_dict = ResMerger.get_exp2_kmnc_suffix_dict()
        dsc_suffix_dict = ResMerger.get_exp2_dsc_suffix_dict()
        for key, v_arr in model_data.items():
            suffix_arr = suffix_dict[key]
            kmnc_suffix_arr = kmnc_suffix_dict[key]
            dsc_suffix_arr = dsc_suffix_dict[key]
            for v, suffix, kmnc_suffix, dsc_suffix in zip(v_arr, suffix_arr, kmnc_suffix_arr, dsc_suffix_arr):
                pair_name = model_conf.get_pair_name(key, v)
                # print(to_base_path,pair_name)
                os.makedirs(os.path.join(to_base_path, pair_name), exist_ok=True)
                # file_name = model_conf.get_pair_name(k, v) + "_imp_val.csv"

                len_X = MyTable.get_exp2_size(key, key + "_harder")
                for k in MyTable(None).exp2_k_list2:
                    df_retrain_arr = []
                    df_idx_arr = []
                    df_ps_arr = []
                    select_size = int(len_X * k)
                    ps_merge_flag = True  # 合并ps
                    for s in [suffix, dsc_suffix, kmnc_suffix]:
                        if s is None:
                            continue
                        dir_path = get_dir_path(s)
                        # 处理retrain
                        fp_retrain = os.path.join(dir_path, f_retrain.format(select_size))
                        df_retrain = pd.read_csv(fp_retrain)
                        df_retrain_arr.append(df_retrain)
                        # 处理idx
                        fp_idx = os.path.join(dir_path, f_idx.format(select_size))
                        df_idx = pd.read_csv(fp_idx)
                        df_idx_arr.append(df_idx)
                        # 处理ps
                        if ps_merge_flag:
                            fp_ps = os.path.join(dir_path, f_ps)
                            df_ps = pd.read_csv(fp_ps)
                            if s == suffix:
                                df_ps_arr.append(df_ps)
                            else:
                                # 只放第0行
                                # print(df_ps)
                                df_ps_arr.append(df_ps.iloc[[0]])
                    # 合并retrain
                    df_merge_retrain = pd.concat(df_retrain_arr)  # 按行合并retrain
                    to_fp_retrain = get_to_path(to_base_path, pair_name, f_retrain.format(select_size))
                    df_merge_retrain.to_csv(to_fp_retrain, index=False)
                    # 合并idx
                    df_merge_idx = pd.concat(df_idx_arr, axis=1)  # 按列合并idx
                    to_fp_idx = get_to_path(to_base_path, pair_name, f_idx.format(select_size))
                    df_merge_idx.to_csv(to_fp_idx, index=False)
                    # 合并ps
                    if ps_merge_flag:
                        df_merge_ps = pd.concat(df_ps_arr)  # 按行合并ps
                        to_fp_ps = get_to_path(to_base_path, pair_name, f_ps)
                        df_merge_ps.to_csv(to_fp_ps, index=False)
                        ps_merge_flag = False

    # 绘制实验2图
    # plot exp2 figs
    def plot_exp2(self, gen_table=True, is_emtpy=False, is_single=False, is_all=False):

        def table(k_list, len_X_select, base_path, imp_val_csv_path, imp_ori_csv_path):
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

        # 自定义图例顺序
        def get_order_legend(handles, labels):
            str_list = ["Random", "Div"]
            for s in str_list:
                for i, label in enumerate(labels):
                    if str(label).find(s) > -1:
                        h = handles[i]
                        handles.remove(h)
                        handles.append(h)
                        l = labels[i]
                        labels.remove(l)
                        labels.append(l)
            return handles, labels

        # 绘制单图
        def fig(k_list, imp_val_fig_path_arr, imp_ori_fig_path_arr, imp_val_csv_path, imp_ori_csv_path):
            df_imp_val = pd.read_csv(imp_val_csv_path)
            df_imp_ori = pd.read_csv(imp_ori_csv_path)
            # 画图
            x_sticks = range(len(k_list))
            x_ticklabels = [str(int(k * 100)) for k in k_list]
            x_ticklabels.insert(0, 0)
            df_arr = [df_imp_val, df_imp_ori]
            path_arr = [imp_val_fig_path_arr, imp_ori_fig_path_arr]
            title_arr = ["imp_val", "imp_ori"]
            for df, p_arr, title in zip(df_arr, path_arr, title_arr):
                for index, row in df.iterrows():
                    label = row["name"] + "_" + row["method"]
                    data = row[3:]
                    if "Div_select" in label:
                        plt.plot(x_sticks, data, label="Div_select", color="crimson")
                    elif "DeepGini" in label:
                        # plt.plot(x_sticks, data, label=label, color="darkgreen")
                        continue
                    elif "Random" in label:
                        plt.plot(x_sticks, data, label="Random", color="black")
                    else:
                        plt.plot(x_sticks, data, label=label, alpha=0.5)
                ax = plt.gca()
                ax.set_xticklabels(x_ticklabels)

                #    绘制一个图例

                if not is_emtpy:
                    plt.xlabel("percentage of test cases")
                    plt.ylabel("acc_imp")
                    plt.title(title)
                    handles, labels = ax.get_legend_handles_labels()  # 获得图例
                    handles, labels = get_order_legend(handles, labels)  # 获得排序后图例
                    plt.legend(handles, labels, loc='upper left', handlelength=1.5, prop={"size": 10}, markerscale=2,
                               labelspacing=0.5)
                    # plt.legend(handles, labels, loc='upper left', )
                for p in p_arr:
                    plt.savefig(p)
                plt.close()

                # 绘制组合图

        res_path = "merge_res/exp2"

        def plot_single():
            model_data = MyTable.get_order_model_data()
            k_list = MyTable(None).exp2_k_list2
            # plot single
            for key, v_arr in model_data.items():
                for v in v_arr:
                    len_X_select = MyTable.get_exp2_size(key, key + "_harder")
                    pair_name = model_conf.get_pair_name(key, v)
                    base_path = os.path.join(res_path, pair_name)

                    table_path = base_path + "/table"
                    fig_path = base_path + "/fig"
                    if is_emtpy:
                        fig_path = fig_path + "/emtpy"
                    os.makedirs(table_path, exist_ok=True)
                    os.makedirs(fig_path, exist_ok=True)

                    imp_val_csv_path = table_path + "/{}_{}_{}.csv".format(key, v, "imp_val")
                    imp_ori_csv_path = table_path + "/{}_{}_{}.csv".format(key, v, "imp_ori")
                    imp_val_fig_path_arr = [fig_path + "/{}_{}_{}.{}".format(key, v, "imp_val", "png"),
                                            fig_path + "/{}_{}_{}.{}".format(key, v, "imp_val", "pdf")]
                    imp_ori_fig_path_arr = [fig_path + "/{}_{}_{}.{}".format(key, v, "imp_ori", "png"),
                                            fig_path + "/{}_{}_{}.{}".format(key, v, "imp_ori", "pdf")]
                    if gen_table:
                        table(k_list, len_X_select, base_path, imp_val_csv_path, imp_ori_csv_path)
                    fig(k_list, imp_val_fig_path_arr, imp_ori_fig_path_arr, imp_val_csv_path, imp_ori_csv_path)

        def plot_all():
            # plot all
            fig_path = res_path + "/fig_all"
            # fig_path = "gen_table"
            imp_val_fig_path_arr = [fig_path + "/{}.{}".format("imp_val", "png"),
                                    fig_path + "/{}.{}".format("imp_val", "pdf")]
            path_arr = imp_val_fig_path_arr
            model_data = MyTable.get_order_model_data()
            k_list = MyTable(None).exp2_k_list2
            x_sticks = range(len(k_list))
            x_ticklabels = [str(int(k * 100)) for k in k_list]
            x_ticklabels.insert(0, 0)

            f = plt.figure(figsize=(20, 20))
            fig_idx = 1
            handles = None
            labels = None
            row_num = 3
            col_num = 3
            for key, v_arr in model_data.items():
                for v in v_arr:
                    # prepare
                    pair_name = model_conf.get_pair_name(key, v)
                    base_path = os.path.join(res_path, pair_name)
                    table_path = base_path + "/table"
                    os.makedirs(fig_path, exist_ok=True)
                    imp_val_csv_path = table_path + "/{}_{}_{}.csv".format(key, v, "imp_val")
                    # plot
                    df_imp_val = pd.read_csv(imp_val_csv_path)
                    # print(row_num, col_num, fig_idx)
                    plt.subplot(row_num, col_num, fig_idx)
                    plt.title(pair_name, fontsize=18)
                    fig_idx += 1
                    for index, row in df_imp_val.iterrows():
                        label = row["name"] + "_" + row["method"]
                        data = row[3:]
                        if "Div_select" in label:
                            plt.plot(x_sticks, data, label="Div_select", color="crimson")
                        elif "DeepGini" in label:
                            # plt.plot(x_sticks, data, label=label, color="darkgreen")
                            continue
                        elif "Random" in label:
                            plt.plot(x_sticks, data, label="Random", color="black")
                        else:
                            plt.plot(x_sticks, data, label=label, alpha=0.5)
                        ax = plt.gca()
                        ax.set_xticklabels(x_ticklabels)
                    # 这里使用mnist LeNet5作为选定legend
                    # 如果指标顺序发生改变,得固定颜色,否则legend的颜色可能对不上
                    if key == model_conf.mnist and v == model_conf.LeNet5:
                        ax = plt.gca()
                        handles, labels = ax.get_legend_handles_labels()
            # 绘制一个图例
            plt.subplot(row_num, col_num, fig_idx)

            handles, labels = get_order_legend(handles, labels)

            plt.legend(handles, labels, loc='center', handlelength=3, handleheight=1, borderaxespad=2, handletextpad=2,
                       fontsize="xx-large")

            # print(handles, labels)
            plt.gca().axis('off')
            # f.legend(handles, labels)  # title="Legend Title" , loc='lower right'
            # f.gca().set_xticklabels("percentage of test cases")
            # f.gca().set_yticklabels("acc_imp")

            f.text(0.5, 0.08, 'percentage of test cases', ha='center', fontsize=20, )
            f.text(0.08, 0.5, 'acc_imp', va='center', rotation='vertical', fontsize=20, )

            # f.ylabel("acc_imp")
            # f.xlabel("percentage of test cases")
            f.suptitle("Acc improve with test cases retrain", fontsize=22, y=0.94)  # y=0.9
            for p in path_arr:
                plt.savefig(p, bbox_inches='tight')
            plt.close("all")

        if is_single:
            plot_single()
        if is_all:
            plot_all()

    # 合并实验1
    # merge exp1 result merge_res/exp1
    def merge_exp1(self):
        def get_dir_path(su):
            base_path = "result"
            dir_name = su + "_" + model_conf.get_pair_name(key, v)
            dir_path = "{}/{}".format(base_path, dir_name)
            return dir_path

        def get_to_path(to_base_path, pair_name, fs):
            return os.path.join(to_base_path, pair_name, fs)

        # 目标文件
        fn = "exp1_{}.csv"
        # 合并
        to_base_path = "merge_res/exp1"
        model_data = MyTable.get_order_model_data()
        suffix_dict = ResMerger.get_exp1_suffix_dict()
        dsc_suffix_dict = ResMerger.get_exp1_dsc_suffix_dict()
        for key, v_arr in model_data.items():
            suffix_arr = suffix_dict[key]
            dsc_suffix_arr = dsc_suffix_dict[key]
            for v, suffix, dsc_suffix in zip(v_arr, suffix_arr, dsc_suffix_arr):
                pair_name = model_conf.get_pair_name(key, v)
                os.makedirs(os.path.join(to_base_path, pair_name), exist_ok=True)
                len_X = MyTable.get_exp1_size(key, key + "_harder")
                for k in MyTable(None).exp1_k_list:
                    df_file_arr = []
                    select_size = int(len_X * k)
                    for s in [suffix, dsc_suffix, ]:
                        if s is None:  # 如果没有dsc则continue
                            continue
                        dir_path = get_dir_path(s)
                        # 处理idx
                        file_path = os.path.join(dir_path, fn.format(select_size))
                        if not os.path.exists(file_path):  # 如果文件不存在则跳过
                            continue
                        df_file = pd.read_csv(file_path)
                        # print(file_path)
                        # print(df_file)
                        # dsc文件
                        if s == dsc_suffix:
                            # 只要两列
                            df_file = df_file[['dsc_time', 'cov_dsc']]
                        df_file_arr.append(df_file)
                    # 合并文件
                    if len(df_file_arr) != 0:
                        df_merge = pd.concat(df_file_arr, axis=1)  # 按列合并df
                        to_fp = get_to_path(to_base_path, pair_name, fn.format(select_size))
                        df_merge.to_csv(to_fp, index=False)

    # 绘制实验1图
    # plot exp1 figs
    def plot_exp1(self, is_all=False):

        # 绘制所有图
        def fig_all(fig_dir, data_name, model_name):
            # data_size
            # base_path
            for k in k_list:
                data_size = int(k * len_X_select)
                csv_path = base_path + "/exp1_{}.csv".format(data_size)  # 3k 6k 3w 6w
                if os.path.exists(csv_path):
                    fig_path = "{}/{}_{}_{}_all.png".format(fig_dir, data_name, model_name, data_size)
                    df = pd.read_csv(csv_path)
                    div_col_arr = ["div", "var", ]
                    cov_col_arr = ["cov_nac", "cov_nbc", "cov_snac", "cov_tknc",
                                   "cov_kmnc", "cov_lsc", "cov_dsc"]
                    col_arr = div_col_arr + cov_col_arr
                    row = 2
                    cr = 4
                    del col_arr[1]
                    fig, axes = plt.subplots(row, cr, figsize=(20, 10))
                    fig.suptitle("{} datasize = {} ".format(data_name, df["data_size"][0]))
                    # size = df["data_size"][0]

                    for i in range(len(col_arr)):
                        col = col_arr[i]
                        if col in df.columns:
                            # x, y = df[col], df["acc_imp"]
                            x, y = df[col], df["wrong_num"]
                            p_res = pearsonr(x, y)
                            sp_res = spearmanr(x, y)
                            r1, p1 = p_res
                            r2, p2 = sp_res
                            # col_name = model_conf.get_pair_name(params["data_name"], params["model_name"])
                            # table.set_df_units(row_name, col_name, values)  # 储存表格
                            if i >= cr:
                                xi = 1
                                yi = i - cr
                            else:
                                xi = 0
                                yi = i
                            axes[xi][yi].set_title("col {} pearsonr={}".format(col, np.round(r1, 2)), )
                            ax = seaborn.jointplot(x=col, y="wrong_num", data=df, kind='reg', color=None,
                                                   ax=axes[xi][yi])
                    fig.savefig(fig_path)
                    plt.close("all")
                    print(fig_path)

        # adjust xticks
        def get_xticks(new_col):
            xticks = None
            if new_col == "NAC":
                xticks = [0.04, 0.08, 0.12, 0.16, 0.20]
            elif new_col == "NBC":
                xticks = [0, 0.05, 0.10, 0.15, 0.20, 0.25]
            elif new_col == "SNAC":
                xticks = [0.00, 0.07, 0.14, 0.21, 0.28]
            elif new_col == "TKNC":
                xticks = [0.78, 0.81, 0.84, 0.87, 0.90]
            elif new_col == "DSC":
                xticks = [0.69, 0.71, 0.73, 0.75, 0.77]
            elif new_col == "LSC":
                xticks = [0.28, 0.29, 0.30, 0.31, 0.32, 0.33]
            return xticks

        # 绘制单图
        def fig_single(fig_dir, ):
            for k in k_list:
                data_size = int(k * len_X_select)
                csv_path = base_path + "/exp1_{}.csv".format(data_size)  # 3k 6k 3w 6w
                if os.path.exists(csv_path):
                    fig_single_dir = os.path.join(fig_dir, "single", str(data_size), )  # 单张图存放的位置
                    os.makedirs(fig_single_dir, exist_ok=True)

                    df = pd.read_csv(csv_path)
                    div_col_arr = ["div", "var", ]
                    cov_col_arr = ["cov_nac", "cov_nbc", "cov_snac", "cov_tknc",
                                   "cov_kmnc", "cov_lsc", "cov_dsc"]
                    col_arr = div_col_arr + cov_col_arr
                    del col_arr[1]

                    for i in range(len(col_arr)):
                        col = col_arr[i]
                        if col in df.columns:
                            # 文件名
                            matplotlib.rcParams.update({'font.size': 12})
                            fig_path = os.path.join(fig_single_dir, col + ".pdf")
                            # 横轴
                            new_col = MyTable.get_table_cols_by_name(col)
                            df[new_col] = df[col]
                            # 纵轴
                            df["Fault detection rate (%)"] = 100 * df["wrong_num"] / data_size
                            # x, y = df[col], df["acc_imp"]
                            x, y = df[new_col], df["Fault detection rate (%)"]

                            p_res = pearsonr(x, y)
                            sp_res = spearmanr(x, y)
                            r1, p1 = p_res
                            r2, p2 = sp_res
                            plt.title("{} pearsonr={}".format(col, np.round(r1, 2)), )
                            ax = seaborn.jointplot(x=new_col, y="Fault detection rate (%)", data=df, kind='reg',
                                                   color=None, )
                            ax.annotate(pearsonr, template='pearson_r: {val:.2f}')

                            if v == model_conf.LeNet1 and key == model_conf.fashion and data_size == 2000:
                                # 设置轴刻度
                                # xlim = ax.fig.gca().get_xlim()
                                xticks = get_xticks(new_col)
                                if xticks is not None:
                                    ax.fig.gca().set_xticks(xticks)
                            plt.savefig(fig_path, bbox_inches='tight')
                            # plt.savefig(fig_path, )
                            plt.close("all")

        res_path = "merge_res/exp1"
        model_data = MyTable.get_order_model_data()
        k_list = MyTable(None).exp1_k_list
        for key, v_arr in model_data.items():
            for v in v_arr:
                len_X_select = MyTable.get_exp1_size(key, key + "_harder")
                pair_name = model_conf.get_pair_name(key, v)
                base_path = os.path.join(res_path, pair_name)
                fig_dir = base_path + "/fig"
                os.makedirs(fig_dir, exist_ok=True)
                if is_all:
                    fig_all(fig_dir, key, v)  # 绘制合并在一起的图,方便观察
                # 绘制单图,用于论文
                fig_single(fig_dir)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    res = ResMerger()

    # res.merge_exp1() # merge your own exp results
    res.plot_exp1()

    # res.merge_exp2() # merge your own exp results
    res.plot_exp2()
