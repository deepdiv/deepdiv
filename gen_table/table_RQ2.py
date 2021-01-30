import pandas as pd
import os

import model_conf
from gen_table.table import MyTable


# 实验2表格
class Table3(MyTable):
    def __init__(self, table_name):
        super().__init__(table_name)
        self.index_arr = MyTable.get_table23_cols()
        self.df_arr = []

    def get_table_data(self):
        base_path = "merge_res/exp2/"
        model_data = MyTable.get_order_model_data()
        for k, v_arr in model_data.items():
            for v in v_arr:
                dir_name = model_conf.get_pair_name(k, v)
                file_name = model_conf.get_pair_name(k, v) + "_imp_val.csv"
                file_path = "{}/{}/table/{}".format(base_path, dir_name, file_name)
                df = pd.read_csv(file_path, index_col=[0])  # 包含了 一个模型,一个数据集,一堆k
                # print(file_name, df.columns)
                for index, row, in df.iterrows():
                    if row["name"] == "DeepGini":
                        continue
                    col_name = MyTable.get_table23_cols_by_name_and_method(row["name"], row["method"])
                    # print(df.columns)
                    for col_k, df_res in zip(self.exp2_k_list, self.df_arr):
                        col_value = row[col_k]
                        # print(col_value)
                        # print(col_name, model_conf.get_pair_name(k, v))
                        # print(self.df.loc[col_name, model_conf.get_pair_name(k, v)])
                        df_res.loc[col_name, model_conf.get_pair_name(k, v)] = col_value
        # 填充T.O.
        for df_res in self.df_arr:
            df_res.loc["KMNC(1000)-CAM"].fillna("T.O.", inplace=True)
        # print(self.df_arr[0])
        # print(self.df.loc[col_name, model_conf.get_pair_name(k, v)], "after")
        #         break
        #     break
        # break

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
    table_name = "table_RQ2.csv"  # 横竖表转换
    table = Table3(table_name)
    table.init_table()
    table.get_table_data()
    table.covert_final_table()
