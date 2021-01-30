import os

import exp1
import exp1_dsc
import exp2
import exp2_dsc
import exp2_kmnc
import model_conf


# mnist
def mnist_LeNet5():
    params = {
        "model_name": model_conf.LeNet5,  # 选取模型
        "data_name": model_conf.mnist,  # 选取数据集
        "nb_classes": model_conf.fig_nb_classes,  # 数据集分类数
        "dau_name": "mnist_harder"  # 选取扩增数据集名称
    }
    return params


def mnist_LeNet1():
    params = {
        "model_name": model_conf.LeNet1,  # 选取模型
        "data_name": model_conf.mnist,  # 选取数据集
        "nb_classes": model_conf.fig_nb_classes,  # 数据集分类数
        "dau_name": "mnist_harder"  # 选取扩增数据集名称
    }
    return params


# fashion
def fashion_LeNet1():
    params = {
        "model_name": model_conf.LeNet1,  # 选取模型
        "data_name": model_conf.fashion,  # 选取数据集
        "nb_classes": model_conf.fig_nb_classes,  # 数据集分类数
        "dau_name": "fashion_harder"  # 选取扩增数据集名称
    }
    return params


def fashion_resNet20():
    params = {
        "model_name": model_conf.resNet20,  # 选取模型
        "data_name": model_conf.fashion,  # 选取数据集
        "nb_classes": model_conf.fig_nb_classes,  # 数据集分类数
        "dau_name": "fashion_harder"  # 选取扩增数据集名称
    }
    return params


# cifar
def cifar_resNet20():
    params = {
        "model_name": model_conf.resNet20,  # 选取模型
        "data_name": model_conf.cifar10,  # 选取数据集
        "nb_classes": model_conf.fig_nb_classes,  # 数据集分类数
        "dau_name": "cifar_harder"  # 选取扩增数据集名称
    }
    return params


def cifar_vgg16():
    params = {
        "model_name": model_conf.vgg16,  # 选取模型
        "data_name": model_conf.cifar10,  # 选取数据集
        "nb_classes": model_conf.fig_nb_classes,  # 数据集分类数
        "dau_name": "cifar_harder"  # 选取扩增数据集名称
    }
    return params


# svhn
def svhn_vgg16():
    params = {
        "model_name": model_conf.vgg16,  # 选取模型
        "data_name": model_conf.svhn,  # 选取数据集
        "nb_classes": model_conf.fig_nb_classes,  # 数据集分类数
        "dau_name": "svhn_harder"  # 选取扩增数据集名称
    }
    return params


def svhn_LeNet5():
    params = {
        "model_name": model_conf.LeNet5,  # 选取模型
        "data_name": model_conf.svhn,  # 选取数据集
        "nb_classes": model_conf.fig_nb_classes,  # 数据集分类数
        "dau_name": "svhn_harder"  # 选取扩增数据集名称
    }
    return params


# exp1
def run_exp1():
    p = mnist_LeNet5()
    exp1.run(p, 0, )
    exp1_dsc.run(p, 0, )

    p = mnist_LeNet1()
    exp1.run(p, 0, )
    exp1_dsc.run(p, 0, )

    p = fashion_LeNet1()
    exp1.run(p, 0, )
    exp1_dsc.run(p, 0, )

    p = fashion_resNet20()
    exp1.run(p, 0, )
    exp1_dsc.run(p, 0, )

    p = cifar_resNet20()
    exp1.run(p, 0, )
    exp1_dsc.run(p, 0, )

    p = cifar_vgg16()
    exp1.run(p, 0, )
    exp1_dsc.run(p, 0, )

    p = svhn_vgg16()
    exp1.run(p, 0, )
    # exp1_dsc.run(p, 0, ) # over time

    p = svhn_LeNet5()
    exp1.run(p, 0, )
    exp1_dsc.run(p, 0, )


# exp2
def run_exp2():
    p = mnist_LeNet5()
    exp2.run(p, 0, )
    exp2_dsc.run(p, 0, )

    p = mnist_LeNet1()
    exp2.run(p, 0, )
    exp2_dsc.run(p, 0, )
    exp2_kmnc.run(p, 0, )

    p = fashion_LeNet1()
    exp2.run(p, 0, )
    exp2_dsc.run(p, 0, )
    exp2_kmnc.run(p, 0, )

    p = fashion_resNet20()
    exp2.run(p, 0, )
    exp2_dsc.run(p, 0, )

    p = cifar_resNet20()
    exp2.run(p, 0, )
    exp2_dsc.run(p, 0, )

    p = cifar_vgg16()
    exp2.run(p, 0, )
    exp2_dsc.run(p, 0, )

    p = svhn_vgg16()
    exp2.run(p, 0, )
    exp2_dsc.run(p, 0, )

    p = svhn_LeNet5()
    exp2.run(p, 0, )
    exp2_dsc.run(p, 0, )


# 准备模型
def prepare_model():
    for k, v_arr in model_conf.model_data:
        for v in v_arr:
            params = {
                "model_name": v,
                "data_name": k,
                "nb_classes": model_conf.fig_nb_classes,
                "dau_name": k + "_harder"
            }
            exp2.set_params(p=params)
            exp2.prepare_model()


if __name__ == '__main__':
    prepare_model()
    run_exp1()
    run_exp2()
