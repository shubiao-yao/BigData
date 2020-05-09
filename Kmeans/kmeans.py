# -*- coding: UTF-8 -*-
import math
import matplotlib.pyplot as plt
import random
import sys


def dd(v):
    print v
    sys.exit(0)

# 欧几里得算法
def getEuclidean(point1, point2):
    dimension = len(point1)
    dist = 0.0
    for i in range(dimension):
        dist += (point1[i] - point2[i]) ** 2
    return math.sqrt(dist)

# kmeans 聚簇算法
def k_means(data_set, k, iteration):
    # 初始化簇心向量，得到数组中三个元素的下标
    index = random.sample(list(range(len(data_set))), k)

    # 得到三个元素对应的 密度和含糖量
    # 例如：
    #      下标： index = [28, 8, 7]
    #      下标为28, 8, 7对应的元素为： dataSet = [[0.725, 0.445], [0.666, 0.091], [0.437, 0.211]]
    vectors = []
    for i in index:
        vectors.append(data_set[i])

    # 初始化标签 写入30个-1
    labels = []
    for i in range(len(data_set)):
        labels.append(-1)

    # 根据迭代次数重复k-means聚类过程
    while (iteration > 0):
        # 初始化簇, 簇的个数是k, 现在初始化为3个簇
        C = []
        for i in range(k):
            C.append([])

        # 遍历全部的元素
        for labelIndex, item in enumerate(data_set):
            classIndex = -1
            # minDist 相当于 1000000.0
            min_dist = 1e6

            # 遍历选中的三个质心，找出当前元素距离最近的簇
            for i, point in enumerate(vectors):
                dist = getEuclidean(item, point)
                if (dist < min_dist):
                    classIndex = i
                    min_dist = dist
            # 得出当前元素距离最近的簇的下标，并将当前元素增加到对应的簇中
            C[classIndex].append(item)
            # 记录当前元素归到哪个簇中
            labels[labelIndex] = classIndex

        # 遍历已经划分好的簇，本例中划分为了3个簇
        for i, cluster in enumerate(C):
            clusterHeart = []

            dimension = len(data_set[0])
            for j in range(dimension):
                clusterHeart.append(0)
            # 遍历其中的某一个簇
            for item in cluster:
                # item：每个簇中的每个元素，记录着西瓜的 密度和含糖量
                for j, coordinate in enumerate(item):
                    # 计算均值 当前元素的密度和含糖量分别除以 当前簇的个数
                    clusterHeart[j] += coordinate / len(cluster)
            vectors[i] = clusterHeart
        iteration -= 1

    return C, labels


# 数据集：每三个是一组分别是西瓜的编号，密度，含糖量
data = """
1,0.697,0.46,2,0.774,0.376,3,0.634,0.264,4,0.608,0.318,5,0.556,0.215,
6,0.403,0.237,7,0.481,0.149,8,0.437,0.211,9,0.666,0.091,10,0.243,0.267,
11,0.245,0.057,12,0.343,0.099,13,0.639,0.161,14,0.657,0.198,15,0.36,0.37,
16,0.593,0.042,17,0.719,0.103,18,0.359,0.188,19,0.339,0.241,20,0.282,0.257,
21,0.748,0.232,22,0.714,0.346,23,0.483,0.312,24,0.478,0.437,25,0.525,0.369,
26,0.751,0.489,27,0.532,0.472,28,0.473,0.376,29,0.725,0.445,30,0.446,0.459"""

# 数据处理 dataSet是30个样本（密度，含糖量）的列表
a = data.split(',')
data_set = [[float(a[i]), float(a[i + 1])] for i in range(1, len(a) - 1, 3)]

# 设置中心个数，这里我们设置为3
k = 3
# 迭代次数
iteration = 10

C, labels = k_means(data_set, k, iteration)

colValue = ['r', 'y', 'g', 'b', 'c', 'k', 'm']
for i in range(len(C)):
    coo_X = []  # x坐标列表
    coo_Y = []  # y坐标列表
    for j in range(len(C[i])):
        coo_X.append(C[i][j][0])
        coo_Y.append(C[i][j][1])
    plt.scatter(coo_X, coo_Y, marker='x', color=colValue[i % len(colValue)], label=i)

# plt.legend(loc='upper right')
plt.show()
print(labels)
