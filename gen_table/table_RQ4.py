import pandas as pd
import os

import model_conf
from gen_table.table import MyTable


# 实验4表格
class Table5(MyTable):
    def __init__(self, table_name):
        super().__init__(table_name)
        # Div_Select 和其他的J距离
        # self.k_list = ["1.0", "5.0", "10.0", "15.0", "20.0", "25.0", "30.0"] #  modify
        # self.k_list = ["1.0", "5.0", "10.0", "20.0", "30.0", "40.0", "50.0"] # ori
        self.df_map = {}
        # index_arr = MyTable.get_table_cols()
        index_arr = ["NAC", "NBC", "SNAC", "TKNC", "KMNC", "LSC", "DSC", "Div"]
        # index_arr.remove("Div_select")
        self.columns = ["Collection", "CAM_Selection", "CAM_% Maxcov", "CAM_# Maxsize", "CTM_Selection", "k=0.1",
                        "k=0.2", "k=0.3"]
        self.index_arr = index_arr

    def get_row_name(self, name):
        if name == "Div_select":
            name = "Div"
        return name

    # 获取表格数据
    def get_table_data(self):
        base_path = "merge_res/exp2/"
        model_data = MyTable.get_order_model_data()
        # 填充cov的数据
        for key, v_arr in model_data.items():
            for v in v_arr:
                pair_name = model_conf.get_pair_name(key, v)
                dir_name = pair_name
                # file_name = model_conf.get_pair_name(k, v) + "_imp_val.csv"
                file_path = "{}/{}/".format(base_path, dir_name)
                # index_arr = np.load(ps_path)
                file_name = "exp_ps_collection.csv"
                fp = os.path.join(file_path, file_name)
                df = pd.read_csv(fp)
                df_res = self.df_map[pair_name]
                for index, row, in df.iterrows():
                    n = self.get_row_name(row["name"])
                    if n == "Div":
                        pass
                    elif n == "DeepGini":
                        pass
                    else:
                        df_res.loc[n, "CAM_% Maxcov"] = MyTable.num_to_str(row["rate"])
                        df_res.loc[n, "Collection"] = int(row["t_collection"])
                        df_res.loc[n, "CAM_Selection"] = int(row["cam_t_selection"])
                        df_res.loc[n, "CAM_# Maxsize"] = row["cam_max"]
                        if n == "LSC" or n == "TKNC" or n == "KMNC" or n == "DSC":
                            df_res.loc[n, "CTM_Selection"] = "N/A"
                        else:
                            df_res.loc[n, "CTM_Selection"] = int(row["ctm_t_selection"])
                # 获取retrain的time
                len_X = self.get_exp2_size(key, key + "_harder")
                t_retrain_arr = []
                for i in range(len(self.exp2_k_list2)):
                    k = self.exp2_k_list2[i]
                    select_size = int(len_X * k)
                    retrain_filename = "test_exp_{}_retrain.csv".format(select_size)
                    rfp = os.path.join(file_path, retrain_filename)
                    df_r = pd.read_csv(rfp)
                    t_retrain = df_r["t_retrain"].mean()
                    t_retrain_arr.append(t_retrain)
                tr_1, tr_2, tr_3 = t_retrain_arr[2], t_retrain_arr[4], t_retrain_arr[6]
                df_res["k=0.1"] = int(tr_1)
                df_res["k=0.2"] = int(tr_2)
                df_res["k=0.3"] = int(tr_3)
                # # 行列都指定 ,多级索引
        for k, df in self.df_map.items():
            # df.loc["KMNC"].fillna("T.O.", inplace=True)
            df.loc["KMNC", "CTM_Selection"] = "N/A"
            for s in ["Collection", "CAM_Selection", "CAM_% Maxcov", "CAM_# Maxsize"]:
                # print(pd.isna(df.loc["KMNC", s]).values)
                if pd.isna(df.loc["KMNC", s]).values:
                    df.loc["KMNC", s] = "T.O."
        # 填充div的数据
        base_path = "result/exp4"
        df_time = pd.read_csv(os.path.join(base_path, "time_cost.csv"))
        for k, df in self.df_map.items():
            fp = os.path.join(base_path, k, "data_select.csv")
            df_select = pd.read_csv(fp)
            rate = MyTable.num_to_str(df_select["div"].mean(), trunc=2)
            max_cov_num = df_select["max_cov"].sum()
            time_c = int(df_time[k].iloc[0])
            df.loc["Div", "CTM_Selection"] = "N/A"
            df.loc["Div", "CAM_Selection"] = time_c
            df.loc["Div", "Collection"] = 0
            df.loc["Div", "CAM_% Maxcov"] = rate
            df.loc["Div", "CAM_# Maxsize"] = max_cov_num
            # print(df)

    def multi_table(self, data):
        up_arr = ["", "CAM", "CAM", "CAM", "CTM", "Retrain", "Retrain", "Retrain", ]
        low_arr = ["Collection", "Selection", "% Maxcov", "# Maxsize", "Selection", "k=0.1", "k=0.2", "k=0.3"]
        df = pd.DataFrame(data, index=self.index_arr, columns=pd.MultiIndex.from_arrays([up_arr, low_arr]))
        return df

    # 非多级索引
    def init_table(self, ):
        model_data = MyTable.get_order_model_data()
        for key, v_arr in model_data.items():
            for v in v_arr:
                name = model_conf.get_pair_name(key, v)
                columns = self.columns
                df = pd.DataFrame(
                    index=self.index_arr,
                    columns=[columns])
                self.df_map[name] = df

    # # 设置单元格的值
    # def set_df_units(self, row_name, col_name, values):
    #     df = self.df
    #     df.loc[row_name, col_name] = values
    #     self.df = df

    # 转换为多级索引
    def covert_final_table(self):
        table_name = str(self.table_name).split(".")[0] + ".xlsx"
        writer = pd.ExcelWriter(os.path.join(self.table_path, table_name))
        for key, df in self.df_map.items():
            multi_df = self.multi_table(df.values)
            multi_df.to_excel(writer, sheet_name=key)
        writer.save()


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    table_name = "table_RQ4.csv"  # 横竖表转换
    table = Table5(table_name)
    table.init_table()
    table.get_table_data()
    table.covert_final_table()
