# 模型设置文件夹
## cifar10
配置文件之间的继承关系为：
 - resnet18 基础配置 resnet18_8xb16_cifar10.py
 - resnet18+hybrid 搜索过程 resnet18_cifar10_seach.py -> 重训练过程 resnet18_cifar10_train.py
## ImageNet
配置文件之间的继承关系为：
 - mobilenetv2 搜索过程 mobilenetv2_search.py

## 实验记录
1. 进化搜索20个resnet18模型，记录其cka，拿去训练看得到acc，计算cka与acc的相关性
```bash
    # 
    bash scripts/hybrid_search/run_hybrid_cifar_resnet18.sh test_search_geatpy_discrete_inference
    bash train_testjson_total_0.sh
    bash train_testjson_total_3.sh
    # 结果放在 test.json 里

```