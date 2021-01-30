import bisect
import multiprocessing
import os
import random
import time
from collections import defaultdict

import numpy as np
import pandas as pd

import cov_config
import data_gener


# cal_d_v only cal data deepdiv cov
# datasets_select  select tagert_size data from test data,there are many situations e.g. class select_status
# get_priority_sequence select data from test data until the end of the algorithm without tagert_size


####################
# 数据点分析
####################
def ck_pq_analyze(ck_list_map, n, i, base_path, S0_i):
    df = None
    csv_data = {}
    for (pp, qq) in get_p_q_list(n, i):
        # 统计并集
        ck_i_list = ck_list_map["{}_{}".format(pp, qq)]
        for idx in range(len(ck_i_list)):
            p, q, a, b, s = ck_i_list[idx]
            vec = S0_i[idx]
            csv_data["i"] = i
            csv_data["p"] = p
            csv_data["q"] = q
            csv_data["idx"] = idx
            csv_data["len"] = b - a
            csv_data["a"] = a
            csv_data["b"] = b
            csv_data["vec_i"] = vec[i]
            csv_data["vec_p"] = vec[p]
            csv_data["vec_q"] = vec[q]
            csv_data["vec_max"] = max(vec)
            csv_data["vec_gini"] = 1 - np.sum(vec ** 2)
            if df is None:  # 如果是空的
                df = pd.DataFrame(csv_data, index=[0])
            else:
                df.loc[df.shape[0]] = csv_data

    base_path = base_path + "/ck_point"
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    csv_path = base_path + "/" + "{}_ck_point_norank.csv".format(i)
    df.to_csv(csv_path, index=False)
    csv_path = base_path + "/" + "{}_ck_point_analyze.csv".format(i)
    df = df.sort_values(axis=0, by=["len"], ascending=False)
    df.to_csv(csv_path, index=False)


##########################
# 数据集选取
##########################


# 计算覆盖对map中所覆盖的总长度
def get_cov_pair_map_len(Ck_pq_map, n, i):
    l_total = 0
    for (p, q) in get_p_q_list(n, i):
        CK_pq = Ck_pq_map["{}_{}".format(p, q)]
        if len(CK_pq) != 0:
            l = np.array(CK_pq)[:, 4].sum()
            l_total += l
    return l_total


# 将新的覆盖对放入SelectMap中
def union_cov_maps(Cx, Cr_select_i_map, X_no_i):
    res_map = {}
    for x_pq in Cx:
        insert_falg = True
        [p, q, a_ins, b_ins, _] = x_pq
        CK_pq = Cr_select_i_map["{}_{}".format(p, q)].copy()
        # 如果CK_pq为空,则直接插入
        if len(CK_pq) == 0:
            res_map["{}_{}".format(p, q)] = CK_pq
        else:
            for ix in range(len(CK_pq)):
                [_, _, a_cur, b_cur, _] = CK_pq[ix]
                # 如果改点的左端点比现有的左端点大,就继续向后找
                if a_ins > a_cur:
                    continue
                # 找到第一个比改点小或者等于的点
                else:
                    CK_pq.insert(ix, x_pq)
                    insert_falg = False
                    break
        if insert_falg:
            CK_pq.append(x_pq)
        res_map["{}_{}".format(p, q)] = CK_pq
    return res_map


class select_status(object):

    def __init__(self) -> None:
        self.SELECT_ALL_LABLES = 0  # 选择了该标签下所有数据
        self.SELECT_ZERO_LABLES = 1  # 该标签有数据,但Smid中没有数据
        self.SELECT_ALL_MID_LABLES = 2  # 该标签有数据,Smid中数据刚好全部被选中
        self.SELECT_ALL_MID_LABLES_WITH_HIGH = 3  # 该标签有数据,Smid中数据全部被选中后也不够,需要补选置信度高的其他数据
        self.SELECT_MID_LABLES_CAM = 4  # 该标签有数据,且Smid中数据全部选中后没有达到最大覆盖
        self.SELECT_MID_LABLES_CAM_ALL = 5  # 该标签有数据,且Smid中数据全部选中后刚好达到最大覆盖
        self.SELECT_MID_LABLES_CAM_CTM = 6  # 该标签有数据,且Smid中数据没选完就达到了最大覆盖,需要使用CTM方式选取Smid中数据


def assert_Tx(Tx_i, target_size):
    if Tx_i.size == 0:
        raise ValueError("该lable下没有数据")  # 应该避免
    if len(Tx_i) < target_size:
        raise ValueError("该lable下的数据不够选")  # 应该避免


def assert_S_mid(S_mid, target_size, symbol, idx_up, idx_mid):
    select_lable = None
    select_idx = []
    s_mid_len = len(S_mid)
    idx_mid = list(idx_mid)
    if s_mid_len == target_size:  # 目标分块里就这么多数据
        select_lable = symbol.SELECT_ALL_MID_LABLES
        select_idx = idx_mid
    elif s_mid_len < target_size:  # 数据不够了,S_mid全选,再从S_up里选点过来
        select_lable = symbol.SELECT_ALL_MID_LABLES_WITH_HIGH
        select_idx += idx_mid
        diff_size = target_size - s_mid_len
        idx_up_ran = list(np.random.randint(len(idx_up), size=diff_size))  # 随机选取idx行
        select_idx += idx_up_ran
        if len(select_idx) != target_size:
            raise ValueError("do not slect target_size points from S_up ")  # 没有从S_up里选出正确个数的点"
    elif s_mid_len == 0:  # 目标分块里没有数据,全从S_up里选
        select_lable = symbol.SELECT_ZERO_LABLES
        diff_size = target_size - s_mid_len
        idx_up_ran = list(np.random.randint(len(idx_up), size=diff_size))  # 随机选取idx行
        select_idx += idx_up_ran
        raise ValueError("taget region does not have points")  # 实验中如果分块里没有数据就抛异常 该分块下没有数据
    return select_lable, select_idx


# 分析覆盖率增长变化
# 排序表
def datasets_select_detail(symbol, S_mid, n, i, target_size, idx_mid, base_path=None, is_analyze=False):
    temp_c_arr = []
    max_cov_point_size = target_size  # 最大覆盖个数
    rel_mid_idx = []  # 使用idx_mid的相对编号
    C_select_i_map = defaultdict(list)
    X_no_i = []
    select_lable = symbol.SELECT_MID_LABLES_CAM
    # 按i p q 获取覆盖对 ipq a b
    ck_list_map = get_ck_list_map(S_mid, n, i)
    ### 将 get_ck_list_map reshpe一下
    C_Xr_list = list(map(list, zip(*ck_list_map.values())))  # 行smaples 列pq_nums 每行为该用例x对应的在所有pq维度下的覆盖对
    if len(idx_mid) != len(C_Xr_list):
        raise ValueError("len idx not eq len data")  # 下标个数与数据个数不符
    ### step1 记录Ctm表格
    # 1.记录每个用例的下标及其最大覆盖长度
    all_rank_arr = []
    for ii, xi in enumerate(C_Xr_list):
        l = np.array(xi)[:, 4].sum()
        all_rank_arr.append([ii, l])
    # 2. 对覆盖长度排序
    all_rank_arr.sort(key=lambda x: (x[1]), reverse=False)  # 由小到大排序
    # 3. 获得排序后的下标及覆盖长度
    all_rank_idx = np.array(all_rank_arr)[:, 0].astype("int").tolist()
    all_rank_cov_len = np.array(all_rank_arr)[:, 1].tolist()
    # if is_analyze:
    #     df = pd.DataFrame({"idx": all_rank_idx, "len": all_rank_cov_len})
    #     df.to_csv(base_path + "/{}_data_rank.csv".format(i))
    #     return
    # # 4.将原有的数据按照覆盖长度的大小重新排序
    # C_Xr_list = np.array(C_Xr_list)[all_rank_idx]
    # # 5. 将第一个点加入到集合中,开始算法
    # rel_mid_idx.append(all_rank_idx[0])  # 添加该数据编号
    # X_no_i.append(all_rank_idx[0])
    # 点选择
    while len(rel_mid_idx) < target_size:
        max_s_i = 0  # 最大覆盖长度
        max_idx = 0  # 最大覆盖长度元素的下标
        max_s_diff = 0
        max_Cr_select_i_map = None  # 最大覆盖集合C
        # step 1 计算选择集的覆盖长度
        s_c_select = get_cov_pair_map_len(C_select_i_map, n, i)  # 当前s_select的覆盖长度
        # 计算覆盖率
        c = get_cov_c(s_c_select, n)
        temp_c_arr.append(c)
        # print("# 当前覆盖率: {}".format(c), time.time())
        # print("# 当前覆盖长: {}".format(s_c_select), time.time())
        # print("# current no select data point", len(X_no_i))
        # 要插入排序的用例
        all_rank_cov_len_copy = all_rank_cov_len.copy()
        all_rank_idx_copy = all_rank_idx.copy()
        for iix in range(len(all_rank_idx) - 1, -1, -1):  # 由大到小遍历
            j = all_rank_idx[iix]  # 当前所选数据编号  iix是给排序后的数组用的, j是给原数组用的, j是用例编号,iix是排序后顺序
            if j in X_no_i:
                continue
            Cx = C_Xr_list[j]  # 当前数据
            Cx_insert = union_cov_maps(Cx, C_select_i_map, X_no_i)
            # step 3 计算并集
            # print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^2", time.time())
            Cx_union = statistic_union_map(Cx_insert, n, i)
            # print(Cx_union, "===")
            # step 3 计算并集的覆盖长度
            # print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^3", time.time())
            s_c_union = get_cov_pair_map_len(Cx_union, n, i)
            # print(s_c_union, "===")
            # step 4 剔除加入后没有变化的数据x
            s_diff = s_c_union - s_c_select  # 覆盖长度差值
            if abs(s_diff) <= 0.001:  # 浮点数计算,如果没有大于一定阈值,认为是没变化
                X_no_i.append(j)
            elif s_c_union > s_c_select:  # 正常流程,即合并后大于上一次的值
                # 如果大于最大值,则更新最大值
                if s_c_union > max_s_i:
                    max_s_i = s_c_union  # 更新最大覆盖长度
                    max_idx = j  # 更新所选数据
                    max_Cr_select_i_map = Cx_union  # 更新C_select
                    max_s_diff = s_diff
                # step5 提前终止算法
                # 判断此前的最大值,是否大于之前的最大值
                if iix != 0 and max_s_diff >= all_rank_cov_len[iix - 1]:
                    # 如果这次增加的了,比上次增加的还多(上次增加的只能<=这次)
                    # print("selected lables: ", len(rel_mid_idx), "early stop in: ", iix, "cur max_s_diff", max_s_diff)
                    break
                else:
                    # 否则,就要更新排序表
                    all_rank_cov_len_copy.remove(all_rank_cov_len[iix])  # 移除对应的编号 和长度
                    all_rank_idx_copy.remove(j)
                    ins_idx = bisect.bisect(all_rank_cov_len_copy, s_diff)
                    all_rank_idx_copy.insert(ins_idx, j)
                    all_rank_cov_len_copy.insert(ins_idx, s_diff)
                    # print(iix, "--->", ins_idx)
            else:
                X_no_i.append(j)
        if max_s_i != 0:  # 如果本次循环找到了一个新的可以增加覆盖的点
            # print("# max_s_i", max_idx)
            rel_mid_idx.append(max_idx)  # 添加该数据编号
            X_no_i.append(max_idx)  # 该数据被选过
            # print("# X_no_i", len(X_no_i))
            # print(C_select_i_map)
            C_select_i_map = max_Cr_select_i_map.copy()  # 更新添加后的覆盖对集合
            all_rank_idx = all_rank_idx_copy.copy()
            all_rank_cov_len = all_rank_cov_len_copy.copy()
            if max_s_diff < 0.005:
                pass
        else:  # 没找到一个新的点,则循环结束
            if len(X_no_i) != len(S_mid):
                # 已经无法选择点了,但X_no_i没有添加所有的编号
                raise ValueError("no point left but X_no_i does not hav all data points")
            if len(rel_mid_idx) == target_size:  # 刚好选择完了所有点
                select_lable = symbol.SELECT_MID_LABLES_CAM_ALL
                max_cov_point_size = len(rel_mid_idx)
                break
            else:  # 补点
                add_num = target_size - len(rel_mid_idx)
                select_lable = symbol.SELECT_MID_LABLES_CAM_CTM
                # 记录最大选择点的数量
                max_cov_point_size = len(rel_mid_idx)
                print("cam max ={}", format(len(rel_mid_idx)))
                Xr_ctm_idx = list(set(X_no_i) - set(rel_mid_idx))  # 没选过,但又被淘汰了的数据编号
                C_Xr_CTM_list = np.array(C_Xr_list)[Xr_ctm_idx]  # 剩下的这些数据里,选覆盖大的
                # 排序 C_Xr_CTM_list, 把每行第4个数加起来后排序
                sorted_arr = []
                for i_pq, x_pq in zip(Xr_ctm_idx, C_Xr_CTM_list):
                    len_s = np.array(x_pq)[:, 4].sum()
                    sorted_arr.append([i_pq, len_s])
                sorted_arr.sort(key=lambda x: (x[1]), reverse=True)  # 由大到小排序

                select_ctm_x = np.array(sorted_arr)[:, 0].astype("int").tolist()
                select_ctm_x = select_ctm_x[:add_num]  # 选择下标,与补点数
                rel_mid_idx += select_ctm_x  # 合并至原始下标内
                # print(rel_mid_idx)
                break
    if is_analyze:
        csv_dir = base_path + "/data_select_profile/"
        os.makedirs(csv_dir, exist_ok=True)
        csv_path = csv_dir + "{}.csv".format(i)
        df = pd.DataFrame(temp_c_arr)
        df.to_csv(csv_path)
    idx_mid = np.array(idx_mid)
    Xr_select_i = idx_mid[rel_mid_idx]  # 根据相对编号获得原始数据的编号
    # np.save(base_path + "/mid_rel_idx_{}.npy".format(i), S_mid[rel_mid_idx[-1]])
    # np.save(base_path + "/{}.npy".format(i), rel_mid_idx)
    if len(Xr_select_i) != target_size:
        raise ValueError("data points size not eq target_size")  # "没有选取到target_size个数的点"
    if len(Xr_select_i) != len(set(Xr_select_i)):
        raise ValueError("some data points  repeatly select")  # 有数据点被重复选取了
    return select_lable, Xr_select_i, C_select_i_map, max_cov_point_size


# 数据选择算法
# 适配各种情况(不是所有情况都会执行数据选择算法)
def datasets_select(Tx, Ty, n, M, ori_target_size, extra_size=0, base_path=None, is_analyze=False):
    print(ori_target_size)
    df = None
    csv_path = base_path + "/data_select_{}.csv".format(ori_target_size)
    csv_data = {}
    c_arr = []  # 存储覆盖率
    max_size_arr = []
    Xr_select = []
    select_lable_arr = []
    symbol = select_status()
    for i in range(n):
        if extra_size == 0:
            target_size = ori_target_size
        else:
            target_size = ori_target_size + 1
            extra_size -= 1
        C_select_i_map = None
        max_cov_point_size = -1  # 最大覆盖点个数
        # 返回值,返回一个选择标记和选择数据编号集
        print("i", i)
        csv_data["label"] = i
        Tx_i, Ty_i, T_idx_arr = data_gener.get_data_by_label_with_idx(Tx, Ty, i)  # 返回所有数据的绝对下标

        assert_Tx(Tx_i, target_size)  # 异常流程,标签不够
        Tx_prob_matrixc = M.predict(Tx_i)  # 原始预测向量矩阵
        if len(Tx_i) == target_size:  # 刚刚好
            select_lable = symbol.SELECT_ALL_LABLES
            abs_idx = T_idx_arr
            C_select_i_map = get_ck_list_map(Tx_prob_matrixc, n, i)  # 改成Tx_prob_matrixc
        else:  # 算法流程
            # 选定区域为S_mid
            ##########################################
            # 未分块, 通常S_up为置信度>0.99的,S_mid为置信度<0.99的,S_low为空
            # 返回T_idx_arr的相对下标
            S_up, rel_idx_up, S_mid, rel_idx_mid, S_low, rel_idx_low \
                = split_data_region_with_idx(Tx_prob_matrixc, i, np.array(range(len(Tx_prob_matrixc))))

            print("len(S0_i)", len(S_mid))
            # 异常流程2,S_mid不够
            # 返回T_idx_arr的相对下标
            select_lable, rel_select_idx = assert_S_mid(S_mid, target_size, symbol, rel_idx_up, rel_idx_mid)
            # 正常流程继续
            if select_lable is None:  # 只有运行算法,才能有最大覆盖
                select_lable, rel_select_idx, C_select_i_map, max_cov_point_size \
                    = datasets_select_detail(symbol, S_mid, n, i, target_size, rel_idx_mid,
                                             base_path=base_path,
                                             is_analyze=is_analyze)
                # print(select_lable, )
                # print(rel_select_idx)
                # # print(C_select_i_map)
                # print(max_cov_point_size)
            if C_select_i_map is None:  # 为None 代表补点了
                C_select_i_map = get_ck_list_map(Tx_prob_matrixc[rel_select_idx], n, i)  # 根据T_idx_arr的相对下标拿数据
            abs_idx = T_idx_arr[rel_select_idx]  # 获得绝对下标
        # 计算所选数据集的覆盖率,与覆盖长度
        s_pq_arr = get_cov_length_map(C_select_i_map, n, i, )
        # 计算第i个lable总覆盖长度与覆盖率率
        s, c_i = get_cov_s_and_c(s_pq_arr, n)
        print("覆盖率 ", c_i)
        # 添加结果
        Xr_select.append(abs_idx)  # 所选数据编号
        select_lable_arr.append(select_lable)  # 该lable下的状态
        csv_data["select_lable"] = select_lable
        c_arr.append(c_i)  # 该lable的覆盖率
        csv_data["div"] = c_i
        # 最大覆盖个数:
        max_size_arr.append(max_cov_point_size)
        csv_data["max_cov"] = max_cov_point_size
        if df is None:  # 如果是空的
            df = pd.DataFrame(csv_data, index=[0])
        else:
            df.loc[df.shape[0]] = csv_data
        df.to_csv(csv_path, index=False)
    return Xr_select, select_lable_arr, c_arr, max_size_arr


def get_priority_sequence_detail(S_mid, n, i, idx_mid, use_add=False):
    temp_c_arr = []
    rel_mid_idx = []  # 使用idx_mid的相对编号
    C_select_i_map = defaultdict(list)
    X_no_i = []
    # 按i p q 获取覆盖对 ipq a b
    ck_list_map = get_ck_list_map(S_mid, n, i)
    ### 将 get_ck_list_map reshpe一下
    C_Xr_list = list(map(list, zip(*ck_list_map.values())))  # 行smaples 列pq_nums 每行为该用例x对应的在所有pq维度下的覆盖对
    if len(idx_mid) != len(C_Xr_list):
        raise ValueError("len idx not eq len data")  # "下标个数与数据个数不符"
    ### step1 记录Ctm表格
    # 1.记录每个用例的下标及其最大覆盖长度
    all_rank_arr = []
    for ii, xi in enumerate(C_Xr_list):
        l = np.array(xi)[:, 4].sum()
        all_rank_arr.append([ii, l])
    # 2. 对覆盖长度排序
    all_rank_arr.sort(key=lambda x: (x[1]), reverse=False)  # 由小到大排序
    # 3. 获得排序后的下标及覆盖长度
    all_rank_idx = np.array(all_rank_arr)[:, 0].astype("int").tolist()
    all_rank_cov_len = np.array(all_rank_arr)[:, 1].tolist()
    # 点选择
    print("************************************", time.time())
    while True:
        max_s_i = 0  # 最大覆盖长度
        max_idx = 0  # 最大覆盖长度元素的下标
        max_s_diff = 0
        max_Cr_select_i_map = None  # 最大覆盖集合C
        # step 1 计算选择集的覆盖长度
        s_c_select = get_cov_pair_map_len(C_select_i_map, n, i)  # 当前s_select的覆盖长度
        # 计算覆盖率
        c = get_cov_c(s_c_select, n)
        temp_c_arr.append(c)
        # print("# 当前覆盖率: {}".format(c), time.time())
        # print("# 当前覆盖长: {}".format(s_c_select), time.time())
        # print("# current no select data point", len(X_no_i))
        # 要插入排序的用例
        all_rank_cov_len_copy = all_rank_cov_len.copy()
        all_rank_idx_copy = all_rank_idx.copy()
        for iix in range(len(all_rank_idx) - 1, -1, -1):  # 由大到小遍历
            j = all_rank_idx[iix]  # 当前所选数据编号  iix是给排序后的数组用的, j是给原数组用的, j是用例编号,iix是排序后顺序
            if j in X_no_i:
                continue
            Cx = C_Xr_list[j]  # 当前数据
            Cx_insert = union_cov_maps(Cx, C_select_i_map, X_no_i)
            # step 3 计算并集
            # print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^2", time.time())
            Cx_union = statistic_union_map(Cx_insert, n, i)
            # print(Cx_union, "===")
            # step 3 计算并集的覆盖长度
            # print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^3", time.time())
            s_c_union = get_cov_pair_map_len(Cx_union, n, i)
            # print(s_c_union, "===")
            # step 4 剔除加入后没有变化的数据x
            s_diff = s_c_union - s_c_select  # 覆盖长度差值
            if abs(s_diff) <= 0.001:  # 浮点数计算,如果没有大于一定阈值,认为是没变化
                X_no_i.append(j)
            elif s_c_union > s_c_select:  # 正常流程,即合并后大于上一次的值
                # 如果大于最大值,则更新最大值
                if s_c_union > max_s_i:
                    max_s_i = s_c_union  # 更新最大覆盖长度
                    max_idx = j  # 更新所选数据
                    max_Cr_select_i_map = Cx_union  # 更新C_select
                    max_s_diff = s_diff
                # step5 提前终止算法
                # 判断此前的最大值,是否大于之前的最大值
                if iix != 0 and max_s_diff >= all_rank_cov_len[iix - 1]:
                    # 如果这次增加的了,比上次增加的还多(上次增加的只能<=这次)
                    # print("selected lables: ", len(rel_mid_idx), "early stop in: ", iix, "cur max_s_diff", max_s_diff)
                    break
                else:
                    # 否则,就要更新排序表
                    all_rank_cov_len_copy.remove(all_rank_cov_len[iix])  # 移除对应的编号 和长度
                    all_rank_idx_copy.remove(j)
                    ins_idx = bisect.bisect(all_rank_cov_len_copy, s_diff)
                    all_rank_idx_copy.insert(ins_idx, j)
                    all_rank_cov_len_copy.insert(ins_idx, s_diff)
                    # print(iix, "--->", ins_idx)
            else:
                X_no_i.append(j)
        if max_s_i != 0:  # 如果本次循环找到了一个新的可以增加覆盖的点
            rel_mid_idx.append(max_idx)  # 添加该数据编号
            X_no_i.append(max_idx)  # 该数据被选过
            C_select_i_map = max_Cr_select_i_map.copy()  # 更新添加后的覆盖对集合
            all_rank_idx = all_rank_idx_copy.copy()
            all_rank_cov_len = all_rank_cov_len_copy.copy()
            if max_s_diff < 0.005:
                pass
        else:  # 没找到一个新的点,则循环结束
            if use_add:
                # 记录最大选择点的数量
                Xr_ctm_idx = list(set(X_no_i) - set(rel_mid_idx))  # 没选过,但又被淘汰了的数据编号
                C_Xr_CTM_list = np.array(C_Xr_list)[Xr_ctm_idx]  # 剩下的这些数据里,选覆盖大的
                # 排序 C_Xr_CTM_list, 把每行第4个数加起来后排序
                sorted_arr = []
                for i_pq, x_pq in zip(Xr_ctm_idx, C_Xr_CTM_list):
                    len_s = np.array(x_pq)[:, 4].sum()
                    sorted_arr.append([i_pq, len_s])
                sorted_arr.sort(key=lambda x: (x[1]), reverse=True)  # 由大到小排序

                select_ctm_x = np.array(sorted_arr)[:, 0].astype("int").tolist()
                rel_mid_idx += select_ctm_x  # 合并至原始下标内
                # print(rel_mid_idx)
                break
            else:
                break
    idx_mid = np.array(idx_mid)
    Xr_select_i = idx_mid[rel_mid_idx]  # 根据相对编号获得原始数据的编号
    if len(Xr_select_i) != len(set(Xr_select_i)):
        raise ValueError("some data points  repeatly select")  # "有数据点被重复选取了"
    return Xr_select_i, C_select_i_map


# 没有target_size
# 目的是选取最大覆盖,而非均匀选取
# 获取获取优先级序列,
# 由数据选择算法更改过来
# 只获取优先级序列 和 最大覆盖率
def get_priority_sequence(Tx, Ty, n, M, base_path=None, use_add=False, is_save_ps=False):
    df = None
    csv_data = {}
    c_arr = []  # 存储覆盖率
    max_size_arr = []  # 存储最大覆盖个数
    Xr_select = []  # 存储选择的下标
    csv_path = base_path + "/data_select.csv"
    ps_path = base_path + "/ps"
    for i in range(n):
        csv_data["label"] = i
        # 返回值,返回一个选择标记和选择数据编号集
        Tx_i, Ty_i, T_idx_arr = data_gener.get_data_by_label_with_idx(Tx, Ty, i)  # 返回所有数据的绝对下标
        Tx_prob_matrixc = M.predict(Tx_i)  # 原始预测向量矩阵
        # 返回T_idx_arr的相对下标
        S_up, rel_idx_up, S_mid, rel_idx_mid, S_low, rel_idx_low \
            = split_data_region_with_idx(Tx_prob_matrixc, i, np.array(range(len(Tx_prob_matrixc))))

        rel_select_idx, C_select_i_map = get_priority_sequence_detail(S_mid, n, i, rel_idx_mid, use_add=use_add)

        # 数据编号
        abs_idx = T_idx_arr[rel_select_idx]
        Xr_select.append(abs_idx)  # 所选数据编号
        csv_data["len(S_up)"] = len(S_up)
        csv_data["len(S_mid)"] = len(S_mid)
        # 覆盖率
        s_pq_arr = get_cov_length_map(C_select_i_map, n, i, )
        # 计算第i个lable总覆盖长度与覆盖率率
        s, c_i = get_cov_s_and_c(s_pq_arr, n)
        print("cov rate ", c_i)
        csv_data["div"] = c_i
        c_arr.append(c_i)  # 该lable的覆盖率

        # 最大覆盖个数:
        max_cov_point_size = len(rel_select_idx)  # 最大覆盖个数
        max_size_arr.append(max_cov_point_size)
        csv_data["max_cov"] = max_cov_point_size
        if df is None:  # 如果是空的
            df = pd.DataFrame(csv_data, index=[0])
        else:
            df.loc[df.shape[0]] = csv_data
        df.to_csv(csv_path, index=False)
    if is_save_ps:
        os.makedirs(ps_path, exist_ok=True)
        for i in range(n):
            idx_arr = Xr_select[i]
            save_path = ps_path + "/{}.npy".format(i)
            np.save(save_path, idx_arr)
    return Xr_select, c_arr, max_size_arr


# 带覆盖率的方差
def cal_d_v(Tx, Ty, n, M, base_path=None, is_anlyze=False, suffix=""):
    c, v, s = cal_d_detail(Tx, Ty, n, M, base_path=base_path, is_anlyze=is_anlyze, suffix=suffix)
    return c, v


# 适配
def cal_d(Tx, Ty, n, M, base_path=None, is_anlyze=False):
    c, v, s = cal_d_detail(Tx, Ty, n, M, base_path=base_path, is_anlyze=is_anlyze)
    return c


# 计算div覆盖率
def cal_d_detail(Tx, Ty, n, M, base_path=None, is_anlyze=False, suffix=""):
    df = None
    c_arr = []  # 存储覆盖率
    S1 = []  # 错误集
    for i in range(n):
        print("i", i)
        csv_data = {}
        Tx_i, Ty_i = data_gener.get_data_by_label(Tx, Ty, i)
        if Tx_i.size == 0:
            # 没有数据则覆盖率为0
            c_arr.append(0)
        else:
            Tx_prob_matrixc = M.predict(Tx_i)  # 原始预测向量矩阵
            # 选定区域为S_mid
            ##########################################
            S_up, S_mid, S_low = split_data_region(Tx_prob_matrixc, i)
            S1.append(S_low)
            print("len(S0_i)", len(S_mid))
            if S_mid.size == 0:  # 目标分块里没有数据
                c_arr.append(0)
                continue
            # 按i p q 获取覆盖对 ipq a b
            ss = time.time()
            ck_list_map = get_ck_list_map(S_mid, n, i)
            ee = time.time()
            # 覆盖对分析
            if is_anlyze:
                ck_pq_analyze(ck_list_map, n, i, base_path, S_mid, )
            sss = time.time()
            # 计算覆盖长度与覆盖率率
            # 计算覆盖长度
            s_pq_arr = get_cov_length_map(ck_list_map, n, i)
            # 计算第i个lable总覆盖长度与覆盖率率
            s, c_i = get_cov_s_and_c(s_pq_arr, n)
            print("覆盖率 ", c_i)
            c_arr.append(c_i)
            eee = time.time()
            # print("cal cov over ..")

            csv_data["label"] = i
            csv_data["数据总量"] = len(Tx_i)
            csv_data["S_up大小"] = len(S_up)
            csv_data["S_low大小"] = len(S_low)
            csv_data["S_mid大小"] = len(S_mid)
            csv_data["covpair_time"] = ee - ss
            csv_data["union_time"] = eee - sss
            csv_data["cov_len"] = s
            csv_data["cov"] = c_i
            if df is None:  # 如果是空的
                df = pd.DataFrame(csv_data, index=[0])
            else:
                df.loc[df.shape[0]] = csv_data
    c = np.array(c_arr).mean()  # 整体覆盖率
    v = np.array(c_arr).var()

    base_path = base_path + "/temp_res"
    os.makedirs(base_path, exist_ok=True)
    csv_path = base_path + "/" + "{}_profile_output.csv".format(suffix)
    if df is not None:
        df.to_csv(csv_path, index=False)
    return c, v, len(np.concatenate(S1, axis=0))


# 将数据分割成 大于上边界的,小于下边界的,在边界之间的
# 在边界间的是重要的
def split_data_region(Tx_prob_matrixc, i):
    Tx_i_prob_vec = Tx_prob_matrixc[:, i]  # 原始模型预测向量矩阵对lablei的预测

    # 小于0.5则该用例可能预测错的 错误:0.1 0.9 正确: 0.4 0.3 0.2
    S1_i = Tx_prob_matrixc[Tx_i_prob_vec < cov_config.boundary]  # 错误集
    # 大于0.5一定是对的,并且要小于0.99, 置信度太高的点1没有意义2会导致数值错误
    # 每一行是一个X_k
    S0_i = Tx_prob_matrixc[(Tx_i_prob_vec >= cov_config.boundary) & (Tx_i_prob_vec < cov_config.up_boundary)]

    # 置信度特别高的点待定
    S2_i = Tx_prob_matrixc[(Tx_i_prob_vec > cov_config.up_boundary)]
    return S2_i, S0_i, S1_i


def split_data_region_with_idx(Tx_prob_matrixc, i, idx):
    Tx_i_prob_vec = Tx_prob_matrixc[:, i]  # 原始模型预测向量矩阵对lablei的预测

    # 小于0.5则该用例可能预测错的 错误:0.1 0.9 正确: 0.4 0.3 0.2
    S1_i = Tx_prob_matrixc[Tx_i_prob_vec < cov_config.boundary]  # 错误集
    idx_1 = idx[Tx_i_prob_vec < cov_config.boundary]
    # 大于0.5一定是对的,并且要小于0.99, 置信度太高的点1没有意义2会导致数值错误
    # 每一行是一个X_k
    S0_i = Tx_prob_matrixc[(Tx_i_prob_vec >= cov_config.boundary) & (Tx_i_prob_vec < cov_config.up_boundary)]
    idx_0 = idx[(Tx_i_prob_vec >= cov_config.boundary) & (Tx_i_prob_vec < cov_config.up_boundary)]

    # 置信度特别高的点待定
    S2_i = Tx_prob_matrixc[(Tx_i_prob_vec > cov_config.up_boundary)]
    idx_2 = idx[(Tx_i_prob_vec > cov_config.up_boundary)]

    return S2_i, idx_2, S0_i, idx_0, S1_i, idx_1


# 获取覆盖对
def get_ck_list_map(S0_i, n, i):
    pq_list = get_p_q_list(n, i)  # 获得pq点集
    ck_map = {}  # 储存覆盖对ck
    # 对x矩阵以方式处理
    for (p, q) in pq_list:
        S0_projection_matrixc = get_projection_matrixc(S0_i, p, q, n, i)
        # S0_projection_matrixc = S0_projection_matrixc[S0_projection_matrixc.min(axis=-1) >= 0]  # 浮点数导致负数计算错误
        # idx = np.argwhere(S0_projection_matrixc.min(axis=-1) >= 0).flatten()
        # S0_projection_matrixc = S0_projection_matrixc[idx]
        i_distance_list = get_i_distance_list(S0_projection_matrixc, i)
        x_k_dot_dot_matrixc = extend_line(S0_projection_matrixc, i)
        ck_i_list = get_cov_pair(i_distance_list, x_k_dot_dot_matrixc, p, q)
        if len(ck_i_list) != 0:  # 浮点数导致的负数值会导致ck_i_list 没元素
            ck_map["{}_{}".format(p, q)] = ck_i_list
        if len(i_distance_list) == len(x_k_dot_dot_matrixc) == len(ck_i_list):
            pass
        else:
            raise ValueError("len ck list  not eq data size")  # ck list 长度不对,不等于用例长度
    return ck_map


# 计算覆盖对
def get_cov_pair(i_distance_list, x_k_dot_dot_matrixc, p, q):
    ck_list = []
    for d, x_k_dot_dot in zip(i_distance_list, x_k_dot_dot_matrixc):
        if d < 0 or x_k_dot_dot[p] < 0:  # 浮点数负数问题
            ck = [p, q, 0, 0, 0]  # 添加加一个不影响覆盖率的值
            ck_list.append(ck)
            continue
        L = get_cov_radius(d)  # 覆盖半径
        a = x_k_dot_dot[p] - L  #
        b = x_k_dot_dot[p] + L
        # print("==", x_k_dot_dot[p], a, b, d, L)
        ##################
        # 对 a,b 进行小数点约束
        ##################
        a = np.round(a, cov_config.round_num)
        b = np.round(b, cov_config.round_num)
        if a < 0:
            a = 0
        if b > 1 - cov_config.boundary:
            b = 1 - cov_config.boundary
        if a > b:
            a = b
        ck = [p, q, a, b, b - a]  # p,q表示对应维度，a,b表示覆盖起始点
        ck_list.append(ck)
    return ck_list


# 计算覆盖长度和覆盖率
def get_cov_s_and_c(s_pq_arr, n):
    # 计算总覆盖长度
    s = np.array(s_pq_arr).sum()
    # 计算覆盖率
    c = get_cov_c(s, n)
    return s, c


# 给长度计算覆盖率
def get_cov_c(s, n):
    c = s / ((1 - cov_config.boundary) * (get_cn2(n)))
    return c


# 统计所有ipq的并集,返回一个map
def statistic_union_map(Ck_pq_map, n, i):
    res_map = {}
    for (p, q) in get_p_q_list(n, i):
        key = "{}_{}".format(p, q)
        CK_pq = Ck_pq_map[key]
        CK_pq = statistic_union(p, q, CK_pq, sort=False)  # 不排序
        res_map[key] = CK_pq
    return res_map


# 统计某个ipq并集,返回一个list
def statistic_union(p, q, Ck_pq_temp, sort=True):
    if len(Ck_pq_temp) == 0:
        return 0
    # 计算 Ck_pq
    Ck_pq = Ck_pq_temp.copy()
    if sort:
        Ck_pq.sort(key=lambda x: (x[2]))  # 即按照a的大小顺序排序
    res = []
    s_pre = Ck_pq[0][2]  # 上一个线段的开头
    e_pre = Ck_pq[0][3]  # 上一个线段的末尾
    for i in range(1, len(Ck_pq)):
        s_cur = Ck_pq[i][2]  # 当前线段的开头
        e_cur = Ck_pq[i][3]  # 当前线段的末尾
        if s_cur <= e_pre:  # 如果当前线段的开头小于上一个线段的末尾
            # 合并两个线段
            e_pre = max(e_cur, e_pre)  # 将两个线段中更长的末尾更新
        else:
            # 出现了一个新的线段
            res.append([p, q, s_pre, e_pre, e_pre - s_pre])  # 将原有线段添加到结果
            s_pre = s_cur
            e_pre = e_cur
    res.append([p, q, s_pre, e_pre, e_pre - s_pre])
    return res


# 计算覆盖长度
def get_cov_length_map(ck_list_map, n, i, ):
    s_pq_arr = []
    pq_list = get_p_q_list(n, i)
    for (p, q) in pq_list:
        # 统计并集
        ck_list = ck_list_map["{}_{}".format(p, q)]
        Ck_pq = statistic_union(p, q, ck_list)
        # 计算覆盖长度s(Ck_pq)
        s_pq = get_cov_length(Ck_pq)
        s_pq_arr.append(s_pq)
        # 计算总覆盖长度
    return s_pq_arr


# 计算覆盖长度
def get_cov_length(Ck_pq):
    total_length = 0
    for i in range(len(Ck_pq)):
        total_length += Ck_pq[i][3] - Ck_pq[i][2]
    return total_length


# 计算覆盖半径
# 保证d越大l越大即可
def get_cov_radius(d):
    if cov_config.is_log:
        l = cov_config.log_ratio * np.log1p(d)
    else:
        l = cov_config.linear_ratio * d
        if cov_config.is_radius_th:
            if d < cov_config.radius_th:
                l = 0
    return l


# 可并行化
def get_i_distance_list(X, i):
    i_distance_list = []
    for x_k_dot in X:
        d = cov_config.up_boundary - x_k_dot[i]  # 不用1减
        i_distance_list.append(d)
    return i_distance_list


#  可并行化
# 延长到交线
def extend_line(X, i):
    x_k_dot_dot_matrixc = X.copy()  # 复制一份矩阵
    n = len(x_k_dot_dot_matrixc[0])  # i的范围 [0,n)
    for x_k_dot in x_k_dot_dot_matrixc:  # 遍历每一个X
        d = 1 - x_k_dot[i]  # 计算 d
        for j in range(n):
            if j == i:
                x_k_dot[j] = cov_config.boundary
                continue
            else:
                x_k_dot[j] = ((1 - cov_config.boundary) / d) * x_k_dot[j]
                # 计算覆盖
    return x_k_dot_dot_matrixc


# X 矩阵,
# 如果该函数计算量大,mars可并行化
def get_projection_matrixc(X, p, q, n, i):
    x_k_dot_matrixc = []
    for x_k in X:
        x_k_dot = get_projection_point(i, p, q, n, x_k)  # 获取投影点
        x_k_dot_matrixc.append(x_k_dot)
    return np.array(x_k_dot_matrixc)


# 求x_k在i,p,q平面的投影点x_k'
# x_k'是n维的,后续只用ipq这三个维度
# def get_projection_point(i, p, q, n, A):
#     # 顶点i
#     P_i = get_vertex(i, n)
#     # 法向量n
#     n = get_normal_vector(i, p, q, n)
#     # 向量P_iA
#     P_iA = A - P_i
#     # 投影向量P_iA_dot
#     P_iA_dot = get_proj(P_iA, n)
#     # 投影点坐标
#     A_dot = P_i + P_iA_dot
#     return A_dot

# 求x_k在i,p,q平面的投影点x_k'
def get_projection_point(i, p, q, n, A):
    one_third = 1 / 3
    two_third = 2 / 3
    A_dot = np.zeros(A.shape)
    A_dot[i] = two_third * A[i] - one_third * A[p] - one_third * A[q] + one_third
    A_dot[p] = two_third * A[p] - one_third * A[q] - one_third * A[i] + one_third
    A_dot[p] = two_third * A[q] - one_third * A[p] - one_third * A[i] + one_third
    return A_dot


# 计算量大可约减
# 如果这里选择全部c92计算量太复杂的话
# 可以选择9个里最大的一个or两个，和剩下8个的组合
def get_p_q_list(n, i):
    num_list = list(range(n))
    num_list.remove(i)
    import itertools
    pq_list = []
    # 抛掉一个点,剩下的排列组合选2个 C92
    for pq in itertools.combinations(num_list, 2):
        pq_list.append(pq)
    return pq_list


##################################################################### 工具函数
# 计算c_n-1^2
def get_cn2(n):
    return (1 / 2 * (n - 1) * (n - 2))


# u在v上投影
def get_proj(u, v):
    v_norm = np.sqrt(sum(v ** 2))
    proj_of_u_on_v = (np.dot(u, v) / v_norm ** 2) * v
    return u - proj_of_u_on_v


# 获取平面顶点 顶点:第i个为1 ,其余为0
# i : [0,n)
def get_vertex(i, n):
    vec = np.zeros((n))
    vec[i] = 1
    return vec


# 获取平面法向量
def get_normal_vector(i, p, q, n, ):
    normal_vec = np.zeros((n))
    normal_vec[i] = 1
    normal_vec[p] = 1
    normal_vec[q] = 1
    return normal_vec
