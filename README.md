# DeepDiv

DeepDiv is a diversity-based criterion for assessing test suites of DNNs.This warehouse is the experimental code of DeepDiv.

![1](https://github.com/deepdiv/deepdiv/blob/master/src/example.jpg)

------

## How to run the code
### environment

```
scikit-learn>=0.19.0
keras>=2.3.1
pandas>=0.23.4
numpy>=1.18.0
tensorflow>=1.13.1
tqdm>=4.23.0
foolbox==1.7.0
seaborn>=0.8.1
matplotlib>=3.0.3
```
run this code to install requirment environment

```
conda install -c fragcolor cuda10.0
conda install cudatoolkit==10.0.130
conda install cudnn==7.6.0

pip install tensorflow-gpu==1.13.1 
pip install keras==2.3.1
pip install numpy==1.18.0
pip install scipy==1.4.1
conda install pandas
conda install scikit-learn
pip install XlsxWriter
pip install seaborn==0.8.1
pip install matplotlib==3.0.3
conda install tqdm
```

### execute

init dirs and data
```
python ./init.py
```

prepare dau
```
python dau_runner.py
```

prepare models and run exp
```
python exp_runner.py
python exp4.py
```

If you want to reproduce our results
We have also prepared the code for tabulation and drawing

first
```
python -m gen_table.res_collection
```

next
```
python -m gen_table.figure_collection
```

finally
```
python -m gen_table.table_RQ1
python -m gen_table.table_RQ2
python -m gen_table.table_RQ3
python -m gen_table.table_RQ4
```

### download model and dau
If you want to get the model and data that we used in our experiment ,you can run the code or
download in this link:

```
https://pan.baidu.com/s/18DKMzs5OlceHSJtYeJKJyQ
ngdh

```


## File code structure

`gen_*` are folder used to generate the data ,model or tables. You can download the model derectly

`exp_*.py` are files contains the experiment

`deepdiv.py` is a file that implements our algorithm

`metrics.py` is a file that implements the neuron coverage

`neural_cov.py` is a file that sorts the data using different coverage methods.

`model_conf.py`  is a file that records model configuration of the neuron coverage measurement file

`SVNH_DatasetUtil.py` is a file  used to load svnh data

`data_gener.py` is a file  used to load data

------

## Data file structure

This data contains 3 folders,`data`,`model`,`adv_image`.
The data we use are mnist dataset, svhn dataset, fashion dataset, cifar10 dataset.

`data`:This folder mainly contains training data and test data of the svhn data set.

`model`:This folder mainly contains the four models we trained in the experiment. The model uses tensorflow as the calculation graph engine and is trained by keras.

`dau`: This folder mainly contains the data-augmentation files. The data structure is ndarray in numpy, which is stored as .npy file using python's pickle method.


## Experimental result

##### Correlation
![1](https://github.com/deepdiv/deepdiv/blob/master/src/RQ1.jpg)

##### Guidance
![2](https://github.com/deepdiv/deepdiv/blob/master/src/RQ2.jpg)
