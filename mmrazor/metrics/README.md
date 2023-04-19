
## 原readme.md
Consult mods.loss for some metrics that are used as apart of the
loss function. It is not clear whether some of these metrics and
metrics used within the loss should be separated yet. It'll become more
clear as I start to structure this repo.
## 一点知识
可以到 scipy, numpy, torch.F 里去找距离函数包括KL散度等
分布之间的距离：
 - Euclidean Distance 欧式距离
 - Manhattan Distance 曼哈顿距离
 - Chebyshev Distance 切比雪夫距离
 - Minkowski Distance 闵可夫斯基距离（一组距离）
   - 当p=1时，为曼哈顿距离；当p=2时，为欧氏距离；当p趋近无穷大时，易证上式即为切比雪夫距离。根据变参数的不同，闵可夫斯基距离可以表示一类距离。
 - Standardized Euclidean distance 标准化欧氏距离
    - 先将各个分量都先进行标准化，再求得标准化后的欧氏距离。
 - Mahalanobis Distance 马氏距离 
   - 若协方差矩阵是单位矩阵（各个样本向量之间独立同分布），则马氏距离就是欧式距离；若协方差矩阵是对角矩阵，则马氏距离就是标准化欧式距离。
 - Lance Williams Distance 兰氏距离
 - Cosine 夹角余弦
 - Tanimoto Coefficient

相关系数：

 - Pearson Correlation Coefficient 皮尔逊相关系数

集合论：

 - Jaccard similarity coefficient 杰卡德相似系数/距离
 
散度：

 - Kullback-Leibler Divergence
 - Jensen-Shannon Divergence
 - Wasserstein Distance = earth mover's Distance
 - Sinkhorn distance

最优传输问题
https://arxiv.org/abs/1209.1077
https://arxiv.org/abs/2003.00855
https://www.microsoft.com/en-us/research/blog/measuring-dataset-similarity-using-optimal-transport/

子模型与原模型/超网之间的变换关系是否问题？
同一个模型每次训练后的权重是否相似？
每轮训练之间的变化哪里最大？

Reference [Links](https://blog.csdn.net/weixin_36670529/article/details/114552770)
## metrics说明
```bash
.
├── accuracy.py # 准确率
├── covariance.py # 计算相关性/协方差
        # cov 计算m自己的协方差
        # cov_norm m,y先归一化再计算协方差
        # cov_eig m,y先归一化并算奇异值，再计算差值
        # cov_eig_kl 
        # cov_kl
├── dist.py
├── emd.py # 空的
├── entropy.py
├── entropy_torch.py
├── exact_mi.py
├── getter.py # Fusion 用到的distance
├── kernel.py
├── mutual_info.py # 互信息
        # entropy (n_samples, n_features)
        # mutual_information (n_samples, n_features)
        # mutual_information_2d 1D array, 1D array
├── README.md
└── sif.py
```