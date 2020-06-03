# -*- coding: UTF-8 -*-
import numpy as np


def get_data():
    data_arr = np.array([[1, 1.1], [1, 1], [0, 0], [0, 0.1]])
    label_arr = ['A', 'A', 'B', 'B']
    return data_arr, label_arr


def knn(point, data_arr, labels, k):
    # 得出训练数据的矩阵，这里的例子是4行2列的矩阵 (m = 4, n = 2)
    m, n = data_arr.shape

    distances = []
    for i in range(m):
        sum_num = 0
        for j in range(n):
            # 计算欧式距离
            sum_num += (point[j] - data_arr[i][j]) ** 2
        distances.append(sum_num ** 0.5)
    # 欧式距离递增排序
    sort_dist = sorted(distances)

    # k个最近的值所属的类别
    class_count = {}
    for i in range(k):
        # 通过标签的数组下标获取标签的值
        label_value = labels[distances.index(sort_dist[i])]
        # class_count是个字典类型，这里计算标签出现的次数 例如：{'B': 1}
        class_count[label_value] = class_count.get(label_value, 0) + 1

    # 根据出现的次数将标签递减排序，出现次数最多的标签在数组的最前面
    sorted_label = sorted(class_count.items(), key=lambda d: d[1], reverse=True)
    return sorted_label[0][0]


if __name__ == '__main__':
    data, label = get_data()
    test_point = [0, 0.2]
    k_value = 3

    r = knn(test_point, data, label, k_value)
    print(r)
