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

import random

import timeit
import math
from scipy.sparse import coo_matrix, lil_matrix, dok_matrix

# 定义基于物品的系统过滤算法

class ItemBasedCF():
    def __init__(self):
        self.n_sim_item = 20 #相似物品数
        self.n_rec_item = 10 #推荐物品数
        self.item_sim_matrix = dok_matrix((624961, 624961), dtype=np.float32) #相似矩阵
        self.item_popular = {} #每个物品被评价次数
        self.item_count = 0 #物品数
        
    def get_dataset(self, train_dataset):
        self.user_group = train_dataset.groupby('user_id').agg([list])[['item_id', 'rank']]
        self.user_ids = train_dataset['user_id'].astype(int)
        self.item_ids = train_dataset['item_id'].astype(int)
        self.ranks = train_dataset['rank'].astype(int)
        self.user_items = train_dataset.groupby('user_id')['item_id'].agg(list)
        print('Load dataset success!')

    '''
    1.  The first step reads each user and the items they have watched in a loop, 
        and counts the number of times each item has been watched, as well as the 
        total number of items; 
    2.  the second step calculates the matrix C, C[i][j] means that you like item 
        i at the same time and the number of users of j, and consider the penalty for active users; 
    3.  the third step is to calculate the similarity between items according to the formula Pearson correlation coefficient; 
    4.  the fourth step is to normalize.
    '''
    def calc_item_sim(self):
        start = timeit.default_timer()
        for iid in self.item_ids:
            if iid not in self.item_popular:
                self.item_popular[iid] = 0
            self.item_popular[iid] += 1
        self.item_count = len(self.item_popular)
        print('item_popular success!')
        print(self.item_count)

        # sim_vector = self.item_sim_matrix[i1]
        
        for user,items in self.user_items.iteritems():
            # 计算项目数量的倒数的对数值
            log_item_count = 1 / math.log(1 + len(items))
            for i1 in items:
                for i2 in items:
                    if i1 == i2:
                        continue
                    self.item_sim_matrix[i1,i2] += log_item_count
        print('Build co-rated users matrix success!')
        
        # 获取稀疏矩阵的元素迭代器
        item_iterator = self.item_sim_matrix.itemset()
        for (i1, i2), value in item_iterator:
            # 计算相似度值
            similarity = value / math.sqrt(self.item_popular[i1] * self.item_popular[i2])
            # 更新稀疏矩阵的元素
            self.item_sim_matrix[i1, i2] = similarity
        print('Calculate item similarity matrix success!')
        
        # 初始化最大值
        max_w = 0
        # 遍历稀疏矩阵的元素
        for (i1, i2), value in self.item_sim_matrix.itemset():
            # 更新最大值
            if value > max_w:
                max_w = value
            # 计算相似度值
            similarity = value / math.sqrt(self.item_popular[i1] * self.item_popular[i2])
            # 更新稀疏矩阵的元素
            self.item_sim_matrix[i1, i2] = similarity
        # 归一化矩阵
        self.item_sim_matrix /= max_w
        all_time = end - start
        print('Time cost is : %fs' % all_time)
        
    #针对目标用户U，找到K个相似的物品，并推荐N个物品，如果用户评价过该物品则不推荐
    def recommend(self, user):
        K = self.n_sim_item
        N = self.n_rec_item
        rank = {}
        used_items = self.user_items[user]
        for item, rating in used_items.items():
            for related_item, w in sorted(self.item_sim_matrix[item].items(), key=itemgetter(1), reverse=True)[:K]:
                if related_item in used_items:
                    continue
                rank.setdefault(related_item,0)
                rank[related_item] += w*float(rating)
        return sorted(rank.items(), key=itemgetter(1),reverse=True)[0:N]
    

itemCF = ItemBasedCF()
itemCF.get_dataset(train_dataset)
itemCF.calc_item_sim()
itemCF.recommend(0)
# itemCF.evaluate()