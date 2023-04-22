# Copyright (c) OpenMMLab. All rights reserved.
'''
    Merger 神经网络权重融合模块设计思路：
    基本模式: 给定一组多个K维的权重向量
        将此组向量定义为一个K+1维的权重矩阵，其中第一维为batch维，其余维度为权重向量的维度。
        则权重融合的定义是：选择某该K+1维权重矩阵的一个或某几个维度，对其进行压缩
            并基于原矩阵依照某种准则融合得到新的权重矩阵。
        设为(C0,C1,C2,...,Ck) -> (C0,D1,D2,C3,...,Ck)
        其中C0为batch维，Ck为最后一维，D1,D2为新的权重矩阵的维度。
    例如：通道剪枝
        单层卷积的权重向量为：(C_out,C_in,K,K)
        对其out_channels维度进行压缩，得到新的权重矩阵为：(C_new,C_in,K,K)
    例如：层剪枝
        对于一组输入输出维度相同的卷积层，其权重向量为：(N,C_out,C_in,K,K)
        对其batch维度进行压缩，得到新的权重矩阵为：(N_new,C_out,C_in,K,K)
    设计思路：
        Merger模块不需要考虑如何压缩权重矩阵，只需要考虑如何融合权重矩阵?
        Merger模块不需要模型的结构信息？可以抽象出只要是权重矩阵，就可以进行融合？
'''

# 偏最小二乘法：计算高维特征与最终结果的相关性，删除相关性最低的特征
# 定义距离函数：计算特征间相似程度，删除与其他特征相似程度最高的特征
# 定义一组基向量：计算特征与基向量的相关性，删除相关性最低的特征
    # 思路发散： 能否通过某种方式，得到每层的最小基向量组（即在此基向量组下，特征间的相关性最低）？
        # 如果将该层剪枝到少于最小基向量组的维度，会导致特征间的相关性增大，从而大大影响模型的精度？
        # Acc Drop