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

# 得到相关信息
unique_user_count = train_dataset['user_id'].nunique()
print("用户总数: ", unique_user_count)
train_item_count = train_dataset['item_id'].nunique()
print("训练集商品总数: ", train_item_count)
unique_item_count = attr_dataset['item_id'].nunique()
print("商品总数: ", unique_item_count)
noattr_item_set = attr_dataset[(attr_dataset['attribute_1']==0) & (attr_dataset['attribute_2']==0)]
noattr_item_count = len(noattr_item_set)
print("无属性商品总数: ", noattr_item_count)
train_records_count = len(train_dataset)
print("训练集评分总数: ", train_records_count)
test_records_count = len(test_dataset)
print("测试集评分总数: ", test_records_count)
test_rank_avg = train_dataset['rank'].sum() / train_records_count
print("训练集评分平均: ", test_rank_avg)
attr_item1_avg = attr_dataset['attribute_1'].sum() / unique_item_count
attr_item2_avg = attr_dataset['attribute_2'].sum() / unique_item_count
print("商品属性1的平均: ", attr_item1_avg)
print("商品属性2的平均: ", attr_item2_avg)

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.sparse import coo_matrix
import math

'''
BiasSvd Model
'''

import random

class BiasSvd(object):
    def __init__(self, alpha, reg_p, reg_q, reg_bu, reg_bi, number_LatentFactors=10, number_epochs=10, columns=["user_id", "item_id", "rank"]):
        self.alpha = alpha # 学习率
        self.reg_p = reg_p
        self.reg_q = reg_q
        self.reg_bu = reg_bu
        self.reg_bi = reg_bi
        self.number_LatentFactors = number_LatentFactors  # 隐式类别数量
        self.number_epochs = number_epochs
        self.columns = columns
        self.P = None
        self.Q = None

    def train(self, dataset):
        '''
        fit dataset
        :param dataset: uid, iid, rating
        :return:
        '''

        self.dataset = pd.DataFrame(dataset)
        self.users_ratings = dataset.groupby(self.columns[0]).agg([list])[[self.columns[1], self.columns[2]]]
        self.items_ratings = dataset.groupby(self.columns[1]).agg([list])[[self.columns[0], self.columns[2]]]
        self.globalMean = self.dataset[self.columns[2]].mean()

        self.P, self.Q, self.bu, self.bi = self.sgd()

    def _init_matrix(self):
        '''
        初始化P和Q矩阵，同时为设置0，1之间的随机值作为初始值
        :return:
        '''
        # User-LF
        P = dict(zip(
            self.users_ratings.index,
            np.random.rand(len(self.users_ratings), self.number_LatentFactors).astype(np.float32)
        ))
        # Item-LF
        Q = dict(zip(
            self.items_ratings.index,
            np.random.rand(len(self.items_ratings), self.number_LatentFactors).astype(np.float32)
        ))
        return P, Q

    def sgd(self):
        '''
        使用随机梯度下降，优化结果
        :return:
        '''
        P, Q = self._init_matrix()

        # 初始化bu、bi的值，全部设为0
        bu = dict(zip(self.users_ratings.index, np.zeros(len(self.users_ratings))))
        bi = dict(zip(self.items_ratings.index, np.zeros(len(self.items_ratings))))

        for i in range(self.number_epochs):
            print("iter %d"%i)
            error_list = []
            for uid, iid, r_ui in self.dataset.itertuples(index=False):
                v_pu = P[uid]
                v_qi = Q[iid]
                err = np.float32(r_ui - self.globalMean - bu[uid] - bi[iid] - np.dot(v_pu, v_qi))

                v_pu += self.alpha * (err * v_qi - self.reg_p * v_pu)
                v_qi += self.alpha * (err * v_pu - self.reg_q * v_qi)

                P[uid] = v_pu 
                Q[iid] = v_qi

                bu[uid] += self.alpha * (err - self.reg_bu * bu[uid])
                bi[iid] += self.alpha * (err - self.reg_bi * bi[iid])

                error_list.append(err ** 2)
            self.alpha = self.alpha * 0.93
            print("mase = ", np.sqrt(np.mean(error_list)))

        return P, Q, bu, bi
    
    def test(self, testset):
        predict_rating = []
        for uid, iid, real_rating in testset.itertuples(index=False):
            try:
                pred_rating = self.predict(uid, iid)
                predict_rating.append(pred_rating)
            except Exception as e:
                print(e)
        result = pd.DataFrame()
        result['user_id'] = testset['user_id']
        result['item_id'] = testset['item_id']
        result['pred'] = predict_rating
        result.to_csv('./data/result.csv', index=False)
        print("Save success!")
        
    def predict(self, uid, iid):

        if uid not in self.users_ratings.index or iid not in self.items_ratings.index:
            return self.globalMean

        p_u = self.P[uid]
        q_i = self.Q[iid]

        return self.globalMean + self.bu[uid] + self.bi[iid] + np.dot(p_u, q_i)


bsvd = BiasSvd(0.0003, 0.00001, 0.00001, 0.01, 0.01, 10, 30)
bsvd.train(train_dataset)
bsvd.test(test_dataset)