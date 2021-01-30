import os

import model_conf
from gen_table.table import MyTable


# copy the figs to figure/
# we can get fig by this code

def copy_exp2_fig():
    exp2_to_base_path = "figure/exp2/"
    os.makedirs(exp2_to_base_path, exist_ok=True)
    base_path = "merge_res/exp2/"
    model_data = MyTable.get_order_model_data()
    for key, v_arr in model_data.items():
        for v in v_arr:
            pair_name = model_conf.get_pair_name(key, v)
            # 目标路径
            to_path = exp2_to_base_path + pair_name + "/"
            os.makedirs(to_path, exist_ok=True)
            # 原路径
            dir_name = pair_name
            # file_name = model_conf.get_pair_name(k, v) + "_imp_val.csv"
            file_path = "{}/{}/fig".format(base_path, dir_name)
            file_arr = os.listdir(file_path)
            for f_name in file_arr:
                source_path = file_path + "/" + f_name
                os.system("cp -r {} {}".format(source_path, to_path))
    source_path = os.path.join(base_path, "fig_all")
    os.system("cp -r {} {}".format(source_path, exp2_to_base_path))


def copy_exp1_fig():
    exp1_to_base_path = "figure/exp1/"
    os.makedirs(exp1_to_base_path, exist_ok=True)
    base_path = "merge_res/exp1/"
    model_data = MyTable.get_order_model_data()
    for key, v_arr in model_data.items():
        for v in v_arr:
            pair_name = model_conf.get_pair_name(key, v)
            # 目标路径
            to_path = exp1_to_base_path + pair_name + "/"
            os.makedirs(to_path, exist_ok=True)
            # 原路径
            dir_name = pair_name
            # file_name = model_conf.get_pair_name(k, v) + "_imp_val.csv"
            file_path = "{}/{}/fig".format(base_path, dir_name)
            file_arr = os.listdir(file_path)
            for f_name in file_arr:
                # 文件 或者文件夹
                source_path = file_path + "/" + f_name
                os.system("cp   -r {} {}".format(source_path, to_path))


if __name__ == '__main__':
    copy_exp2_fig()
    copy_exp1_fig()
