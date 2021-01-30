import os


def init_data():
    tran_path = "http://ufldl.stanford.edu/housenumbers/train_32x32.mat"
    test_path = "http://ufldl.stanford.edu/housenumbers/test_32x32.mat"
    # extra_path = "http://ufldl.stanford.edu/housenumbers/extra_32x32.mat"
    train_data_path = "./data/svhn/SVHN_train_32x32.mat"
    test_data_path = "./data/svhn/SVHN_test_32x32.mat"
    # extra_data_path = "./data/svhn/SVHN_extra_32x32.mat"
    if not os.path.exists(train_data_path):
        os.system("curl -o {} {}".format(train_data_path, tran_path))
    if not os.path.exists(test_data_path):
        os.system("curl -o {} {}".format(test_data_path, test_path))
    # if not os.path.exists(extra_data_path):
    #     os.system("curl -o {} {}".format(extra_data_path, extra_path))


def init_dirs():
    if not os.path.exists("./model"):
        os.makedirs("./model")
    if not os.path.exists("./data/svhn"):
        os.makedirs("./data/svhn")
    if not os.path.exists("./result"):
        os.makedirs("./result")
    if not os.path.exists("./table"):
        os.makedirs("./table")
    if not os.path.exists("./figure"):
        os.makedirs("./figure")
    if not os.path.exists("./dau"):
        os.makedirs("./dau")
    if not os.path.exists("./temp_model/temp"):
        os.makedirs("./temp_model/temp")


# def init_pdf_models():
#     os.system("cd pdf")
#     os.system("python pdf_models.py")
#
#
# def init_Drebin_models():
#     os.system("cd Drebin")
#     os.system("python app_models.py")
#
#
# def init_Drebin_data():
#     os.system("cd Drebin")
#     os.system("python data_utils.py")


if __name__ == '__main__':
    init_dirs()
    init_data()
