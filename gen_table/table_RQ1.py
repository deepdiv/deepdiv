import pandas as pd
import os

import model_conf
from gen_table.table import MyTable
from scipy.stats import pearsonr, spearmanr


class Table2(MyTable):
    def __init__(self, table_name):
        super().__init__(table_name)
        self.df_arr = []
        self.index_arr = ["KMNC", "NBC", "TKNC", "NAC", "SNAC", "DSC", "LSC", "Div"]

    def get_table_data(self, ):
        base_path = "merge_res/exp1/"
        model_data = MyTable.get_order_model_data()
        for key, v_arr in model_data.items():
            for v in v_arr:
                dir_name = model_conf.get_pair_name(key, v)
                len_X = MyTable.get_exp1_size(key, key + "_harder")
                for i in range(len(self.exp1_k_list)):
                    df_res = self.df_arr[i]
                    k = self.exp1_k_list[i]
                    select_size = int(len_X * k)
                    file_name = "exp1_{}.csv".format(select_size)
                    file_path = "{}/{}/{}".format(base_path, dir_name, file_name)
                    if os.path.exists(file_path):
                        df = pd.read_csv(file_path, index_col=[0])  # 包含了 一个模型,一个数据集,一堆k
                        div_col_arr = ["div", ]
                        cov_col_arr = ["cov_nac", "cov_nbc", "cov_snac", "cov_tknc",
                                       "cov_kmnc", "cov_lsc", "cov_dsc", ]
                        col_arr = div_col_arr + cov_col_arr
                        # print(col_arr)
                        for col in col_arr:
                            if col == "cov_dsc":
                                if col not in df.columns:  # 没有DSC,就略过
                                    continue
                            x, y = df[col], df["wrong_num"]
                            p_res = pearsonr(x, y)
                            sp_res = spearmanr(x, y)
                            r1, p1 = p_res
                            r2, p2 = sp_res
                            value = "({}/{}) ({}/{})".format(MyTable.num_to_str(r1), MyTable.num_to_str(p1, trunc=3),
                                                             MyTable.num_to_str(r2), MyTable.num_to_str(p2, trunc=3))
                            # print(col)
                            new_col_name = MyTable.get_table_cols_by_name(col)
                            df_res.loc[new_col_name, model_conf.get_pair_name(key, v)] = value

    #
    # # 行列都指定 ,多级索引
    def multi_table(self, data):
        params = MyTable.get_order_model_data()
        key_arr = []
        v_arr = []
        for key, values_arr in params.items():
            key_arr.append(key)
            key_arr.append(key)
            for v in values_arr:
                v_arr.append(v)
        df = pd.DataFrame(data, index=self.index_arr, columns=[key_arr, v_arr])
        return df

    # 非多级索引
    def init_table(self, ):
        for _ in range(len(self.exp1_k_list)):
            pair_list = MyTable.get_order_pair_name()
            df = pd.DataFrame(
                index=self.index_arr,
                columns=[pair_list])
            self.df_arr.append(df)

    # 字体标红
    def high_score_red(self, s, p=0.05):
        color = 'red' if s >= p else 'black'
        return f"color:{color}"

    # 转换为多级索引
    def covert_final_table(self, ):
        table_name = str(self.table_name).split(".")[0] + ".xlsx"
        writer = pd.ExcelWriter(os.path.join(self.table_path, table_name), engine="xlsxwriter")  # 设置引擎
        for df, k in zip(self.df_arr, self.exp1_k_list):
            multi_df = self.multi_table(df.values)
            multi_df.to_excel(writer, sheet_name=str(k))
        writer.save()


if __name__ == '__main__':
    table_name = "table_RQ1.csv"
    table = Table2(table_name)
    table.init_table()
    table.get_table_data()
    table.covert_final_table()
