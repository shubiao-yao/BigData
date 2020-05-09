# -*- coding: UTF-8 -*-
import Knn

# 生成训练样本
group, labels = Knn.createDataSet()
# 对测试数据[0,0]进行KNN算法分类测试
res = Knn.classify([0, 0], group, labels, 3)

print res
