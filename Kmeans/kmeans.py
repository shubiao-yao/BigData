# -*- coding: utf-8 -*-
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt


# 通过欧几里得算法计算两点之间的距离
def get_distance(first_point, second_point):
    return np.sqrt(np.sum(np.square(first_point - second_point)))


# 对每个属于data_arr的item， 计算item与center_point中k个质心的距离，找出距离最小的，并将item加入相应的簇类中
def get_cluster(data_arr, center_point):
    # 使用字典保存聚类后的结果
    cluster_dict = dict()
    # 质心的个数
    k_len = len(center_point)

    for first_point in data_arr:
        flag = -1
        # 初始化最大距离
        min_distance = np.inf
        for i in range(k_len):
            second_point = center_point[i]
            # 计算两点之间的距离
            distance = get_distance(first_point, second_point)
            # 本例子给的质心数是4，内部循环会拿着first_point逐个与4个质心做距离计算，最终会得出最近的距离
            if distance < min_distance:
                # 保存最近的距离
                min_distance = distance
                # 保存距离最近的质心数组下标
                flag = i
        # 如果质心不存在于字典中，在字典中初始化以质心数组下标为key
        if flag not in cluster_dict.keys():
            cluster_dict.setdefault(flag, [])

        # 将当前计算的点加入到对应的簇中
        cluster_dict[flag].append(first_point)
    return cluster_dict


# 重新计算得到k个质心
def get_new_center_point(cluster_dict):
    new_center_point = []
    for key in cluster_dict.keys():
        centroid = np.mean(cluster_dict[key], axis=0)
        new_center_point.append(centroid)
    # 通过聚类后的字典得到新的质心点
    return new_center_point


def get_sum_value(center_point, cluster_dict):
    sum_value = 0.0
    for key in cluster_dict.keys():
        first_point = center_point[key]
        distance = 0.0
        for point in cluster_dict[key]:
            distance += get_distance(first_point, point)
        # 将簇中各个点与质心的距离累加求和
        sum_value += distance
    return sum_value


def show(center_point, cluster_dict):
    # 展示聚类结果
    color_mark = ['or', 'ob', 'og', 'ok', 'oy', 'ow']  # 不同簇类标记，o表示圆形，另一个表示颜色
    center_point_mark = ['dr', 'db', 'dg', 'dk', 'dy', 'dw']

    for key in cluster_dict.keys():
        plt.plot(center_point[key][0], center_point[key][1], center_point_mark[key], markersize=12)  # 质心点
        for item in cluster_dict[key]:
            plt.plot(item[0], item[1], color_mark[key])
    plt.show()


# 从文件中读取数据
def get_data():
    data_set = pd.read_csv("./test.txt")
    return data_set.values


# 从数据集中选取k个初始质心
def init_center_point(data_arr, k):
    return random.sample(data_arr, k)


def k_means(k):
    # 从文件中获取需要聚类的数据
    data_arr = get_data()

    # 从数据集中选取k个初始质心
    center_point = init_center_point(data_arr, k)

    # 首次聚类为4个簇，得出每个簇中对应的点
    cluster_dict = get_cluster(data_arr, center_point)

    sum_value = get_sum_value(center_point, cluster_dict)
    old_value = 1

    cluster_dict_result = dict()
    center_point_result = []
    # 当两次聚类的误差小于某个值时，说明质心已经不再变化
    while abs(sum_value - old_value) >= 0.00001:
        # 重新计算得到k个质心
        center_point_result = get_new_center_point(cluster_dict)
        # 聚类出新的4个簇
        cluster_dict_result = get_cluster(data_arr, center_point_result)

        old_value = sum_value
        sum_value = get_sum_value(center_point_result, cluster_dict_result)

    show(center_point_result, cluster_dict_result)


if __name__ == '__main__':
    # 簇的个数，这里初始化为4
    k_value = 4

    k_means(k_value)
