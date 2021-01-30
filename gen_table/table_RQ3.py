import pandas as pd
import os

import model_conf
from gen_table.table import MyTable


# 实验4表格
class Table4(MyTable):
    def __init__(self, table_name):
        super().__init__(table_name)
        self.df_arr = []
        index_arr = MyTable.get_table23_cols()
        index_arr.remove("Div_select")
        self.index_arr = index_arr

    # 相似度
    @staticmethod
    def get_jaccard(set_a, set_b):
        # import scipy.spatial.distance as dist  # 导入scipy距离公式
        # matV = mat([[1, 1, 0, 1, 0, 1, 0, 0, 1], [0, 1, 1, 0, 0, 0, 1, 1, 1]])
        # print("dist.jaccard:", dist.pdist(matV, 'jaccard')
        unions = len(set_a.union(set_b))
        intersections = len(set_a.intersection(set_b))
        return 1. * intersections / unions

    # 获取表格数据
    def get_table_data(self):
        base_path = "merge_res/exp2/"
        model_data = MyTable.get_order_model_data()
        for key, v_arr in model_data.items():
            for v in v_arr:
                dir_name = model_conf.get_pair_name(key, v)
                file_path = "{}/{}/".format(base_path, dir_name)
                len_X = self.get_exp2_size(key, key + "_harder")
                for i in range(len(self.exp2_k_list2)):
                    k = self.exp2_k_list2[i]
                    df = self.df_arr[i]
                    select_size = int(len_X * k)
                    idx_file_name = "test_exp_{}_idx.csv".format(select_size)
                    idx_path = os.path.join(file_path, idx_file_name)
                    df_idx = pd.read_csv(idx_path)
                    col_name_arr = df_idx.columns
                    div_select_idx_arr = df_idx["Div_select"].values
                    # print(div_select_idx_arr.shape)
                    metrics_method_dict = MyTable.get_metrics_method()
                    del metrics_method_dict["Div_select"]
                    for m_name, m_method_arr in metrics_method_dict.items():
                        for m_method in m_method_arr:
                            if m_name == "DeepGini":
                                continue
                            if m_name == "Random":
                                col_name = m_name
                            else:
                                col_name = m_name + "_" + m_method
                            if col_name in col_name_arr:
                                cov_arr = df_idx[col_name].values
                                J_value = Table4.get_jaccard(set(div_select_idx_arr), set(cov_arr))
                                df.loc[MyTable.get_table23_cols_by_name_and_method(m_name, m_method),
                                       model_conf.get_pair_name(key, v)] = J_value
                            else:
                                pass

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
        for _ in range(len(self.exp2_k_list)):
            pair_list = MyTable.get_order_pair_name()
            df = pd.DataFrame(
                index=self.index_arr,
                columns=[pair_list])
            self.df_arr.append(df)

    # 转换为多级索引
    def covert_final_table(self):
        table_name = str(self.table_name).split(".")[0] + ".xlsx"
        writer = pd.ExcelWriter(os.path.join(self.table_path, table_name))
        for df, k in zip(self.df_arr, self.exp2_k_list):
            multi_df = self.multi_table(df.values)
            multi_df.to_excel(writer, sheet_name=str(float(k) * 0.01))
        writer.save()


if __name__ == '__main__':
    table_name = "table_RQ3.csv"  # 横竖表转换
    table = Table4(table_name)
    table.init_table()
    table.get_table_data()
    table.covert_final_table()
    table.covert_final_table()
    # print(table.multi_table(None).columns.values)
    # print(table.df_arr[0].columns.values)
