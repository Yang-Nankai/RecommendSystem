###############################################################################
# 重要: 请务必把任务(jobs)中需要保存的文件存放在 results 文件夹内
# Important : Please make sure your files are saved to the 'results' folder
# in your jobs
###############################################################################
attr_file_path = './data/itemAttribute.txt'
test_file_path = './data/test.txt'
train_file_path = './data/train.txt'
attr_csvfile_path = './data/attr.csv'
train_csvfile_path = './data/train.csv'
test_csvfile_path = './data/test.csv'
#将txt文档以csv的形式存储，方便操作

#将itemAttribute转为csv格式
import csv
import pandas as pd
import numpy as np

train_dataset = pd.read_csv(filepath_or_buffer=train_csvfile_path,
                              sep=',', header=None, names=['user_id', 'item_id', 'rank'],
                              skiprows=1)
test_dataset = pd.read_csv(filepath_or_buffer=test_csvfile_path,
                              sep=',', header=None, names=['user_id', 'item_id', 'rank'],
                              skiprows=1)
attr_dataset = pd.read_csv(filepath_or_buffer=attr_csvfile_path,
                              sep=',', header=None, names=['item_id', 'attribute_1', 'attribute_2'],
                              skiprows=1)

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.sparse import coo_matrix
import math
import pickle

'''
BiasSvd Model
'''

import random

columns=["user_id", "item_id", "rank"]
dataset = pd.DataFrame(train_dataset)
users_ratings = dataset.groupby(columns[0]).agg([list])[[columns[1], columns[2]]]
items_ratings = dataset.groupby(columns[1]).agg([list])[[columns[0], columns[2]]]
globalMean = dataset[columns[2]].mean()

# 从文件加载对象
with open('./results/model.pkl', 'rb') as f:
    P, Q, bu, bi = pickle.load(f)

def predict(uid, iid):
    
    uid = int(uid)
    iid = int(iid)

    p_u = P[uid]
    q_i = Q[iid]

    return globalMean + bu[uid] + bi[iid] + np.dot(p_u, q_i)

while True:
    uid = input("uid: ")
    iid = input("iid: ")
    print(predict(uid, iid))