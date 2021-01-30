import os

if __name__ == '__main__':
    os.system("python -m gen_data.dau_mnist")
    os.system("python -m gen_data.dau_fashion")
    os.system("python -m gen_data.dau_cifar")
    os.system("python -m gen_data.dau_svhn")
