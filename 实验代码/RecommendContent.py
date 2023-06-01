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


# 包含所有物品平均分的字典grouped_data
def get_avg():
    '''
    获得平均分
    '''
    avg_data = train_dataset.groupby('item_id')['rank'].mean()
    return avg_data

# 得到存储所有物品种类的字典item_cast
def get_item_cast():
    '''
    得到所有种类
    '''
    attribute_1_values = attr_dataset['attribute_1'].unique().tolist()
    attribute_2_values = attr_dataset['attribute_2'].unique().tolist()
    # 合并两个列表并去重
    item_cast = list(set(attribute_1_values + attribute_2_values))
    return item_cast

# 得到所有种类下排名前100的物品列表
def get_item_sort(avg_data, item_cast):
    '''
    得到种类前100
    '''
    item_sort = {}
    merged_data = pd.merge(avg_data, attr_dataset, on='item_id')
    for attribute in item_cast:
        item_sort[attribute] = []
        # 筛选出该种类的物品，并按照平均分进行降序排序)
        item_sort[attribute].append(merged_data[merged_data['attribute_1'] == attribute].sort_values(by='rank', ascending=False)['item_id'])
        item_sort[attribute].append(merged_data[merged_data['attribute_2'] == attribute].sort_values(by='rank', ascending=False)['item_id'])
    return item_sort

# 得到用户的偏好
def get_user_like():
    '''
    得到用户最喜欢的几个种类
    '''
    user_like = {}
    user_items = train_dataset.groupby('user_id')['item_id'].agg(list)
    user_ratings = train_dataset.groupby('user_id')['rank'].agg(list)
    user_count = len(user_items)
    for uid in range(0, user_count):
        user_like[uid] = {}
        index = 0
        if uid % 1000 == 0:
            print(uid)
        for iid in user_items[uid]:
            #找到物品属于的种类
            attr = attr_dataset.iloc[iid][1:].tolist()
            for a in attr:
                if a not in user_like[uid]:
                    # print(uid, iid)
                    user_like[uid][a] = 0
                user_like[uid][a] += user_ratings[uid][index] * 0.5
            index += 1
    return user_like
    
#对用户进行推荐           
def recom(user_like, user_id, item_sort):
    topk = 2
    # 对用户喜欢的属性进行排序
    for uid in user_id:
        sorted_keys = sorted(user_like[uid], key=user_like[uid].get, reverse=True)
        key = 0
        if len(sorted_keys) != 0:
            if sorted_keys[0] == 0:
                if len(sorted_keys) > 1:
                    key = sorted_keys[1]
                    print("该用户最喜欢种类", sorted_keys[1])
                else:
                    key = sorted_keys[0]
                    print("该用户最喜欢种类", sorted_keys[0])
            else:
                key = sorted_keys[0]
                print("该用户最喜欢种类", sorted_keys[0])
            #然后对其开始推荐
            print("为用户", uid, "进行推荐: ")
            
            for i in range(topk):
                print(item_sort[key][i][:2].values)
        else:
            print("无法推荐!")

avg_data = get_avg()
item_cast = get_item_cast()
item_sort = get_item_sort(avg_data, item_cast)
user_like = get_user_like()
user_id = [0, 56, 89, 1234, 12345]
recom(user_like, user_id, item_sort)