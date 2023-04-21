#  项目记录
基于MMRazor 版本创建
## 项目开发
调用层级：
Scripts -> Main Python + Config -> Algorithm -> Pruner -> Metric

- Scripts: 脚本入口，用于调用Main Python，控制一些超参数和路径，如数据集路径、模型路径、日志路径等
- Main Python: 项目入口，包含加载数据集、模型、优化器、损失函数等，
- Config: 配置文件
- Algorithm: 算法入口，进行流程控制，比如迭代式剪枝、蒸馏中的迭代训练控制以及全局剪枝
- Pruner: 剪枝器，进行剪枝操作、MASK管理
- Metric: 评价指标，用于剪枝和蒸馏中的评价指标计算

若要添加新算法，按上述动作层级创建文件进行开发
关于输出,使用: logger
```python 
    from mmcls.utils import get_root_logger
    # ...
    logger=get_root_logger(log_level='INFO')
    # ...
    logger.info(f'INFO: checkpoints contains keys: {checkpoint.keys()}')

```
## 项目文件结构
``` shell
```
### 关于Pruner

channel_pruning
``` python
ratio # Global pruning ratio
contents={'layer_name':(weight/features)}
metric_scores=pruner.get_metric_local(metric_name,contents)
subnet_dict=pruner.sample_subnet_uniform(ratio,metric_scores) # uniform pruning
subnet_dict=pruner.sample_subnet_diff(ratio,metric_scores) # non-uniform pruning with global comparison
ratios= XXX
subnet_dict=pruner.sample_subnet_nonuni(ratios,metric_scores) # non-uniform pruning with denoted ratio

pruner.set_subnet(subnet_dict) # set net to pruned status
```

**Pruner控制的部分**

- 建立整个模型的结构图
- 根据Algo传入的Metric排序,产生out_mask
  - channel_space: 记录每组共享out_mask的层
- set_subnet: 设置剪枝后网络

**Algorithm控制的部分**

## 迁移后需要修改的内容：

ResNet_Cifar 

mmcls.models.backbones.resnet_cifar
```python
    def __init__(self, depth, deep_stem=False, **kwargs):
        if depth in [20, 32, 44, 56, 110, 1202]:
            kwargs['stem_channels'] = 16
            kwargs['base_channels'] = 16
            kwargs['avg_down'] = True # optional
            kwargs['out_indices'] = (2,)
        super(ResNet_CIFAR, self).__init__(
            depth, deep_stem=deep_stem, **kwargs)
        assert not self.deep_stem, 'ResNet_CIFAR do not support deep_stem'
```
mmcls.models.backbones.resnet
```python
    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3)),
        20: (BasicBlock, [3, 3, 3]),
        32: (BasicBlock, [5, 5, 5]),
        44: (BasicBlock, [7, 7, 7]),
        56: (BasicBlock, [9, 9, 9]),
        110: (BasicBlock, [18, 18, 18]),
        1202: (BasicBlock, [200, 200, 200])
    }
```
<!-- /home/xxxy/miniconda3/envs/wyh_graduate/lib/python3.10/site-packages/mmcls/models/backbones/resnet.py -->
https://github.com/open-mmlab/mmpretrain/issues/801

Pretrained model path https://github.com/open-mmlab/mmpretrain/tree/main/configs/resnet

### 数据集路径
ImageNet 1k 路径
```python
data=dict(
    samples_per_gpu=1024, 
    workers_per_gpu=16,
    train=dict(
        data_prefix='/data/imagenet/ILSVRC2012_img_train',
        ),
    val=dict(
        data_prefix='/data/imagenet/val',
        ann_file=None,
        ),
    test=dict(
        # replace `data/val` with `data/test` for standard test
        data_prefix='/data/imagenet/val',
        ann_file=None,
        )
    )
```
### 模型Checkpoint路径

### log存储路径

## 运行命令
在当前项目路径下： 
示例命令  
```bash
# One-shot 剪枝
bash tests/simple_pruning/run_test_mmcls.sh
# Hybrid-search
bash scripts/hybrid_search/run_hybrid_cifar_resnet18.sh test_search_geatpy_discrete_inference
# Fusion
bash tests/simple_fusion/run_fusion_mmcls.sh
# bash scripts/fusion_merge/run_fusion_cifar_resnet18.sh test_fusion_init
# ImageNet推理

```
可以加速sklearn的Intel CPU 计算
        $ conda install scikit-learn-intelex
        $ python -m sklearnex my_application.py

**普通剪枝**

CDP
```shell
CUDA_VISIBLE_DEVICES=2,3 NCCL_DEBUG=INFO MKL_NUM_THREADS=1 bash tests/simple_pruning/dist_train.sh

# CDP 可视化参数
    bash tests/simple_pruning/CDP/visual_cdp_score_resnet18.sh


```

**搜索剪枝**
2023.04.19

搜索json结果保存到和checkpoint一致的路径中去。

```shell

# Cifar 搜索+训练
bash scripts/hybrid_search/run_hybrid_cifar_resnet18.sh test_search_geatpy_discrete_inference
# ImageNet 搜索+训练
bash scripts/hybrid_search/run_hybrid_cifar_resnet18.sh test_search_geatpy_discrete_inference
# 训练 单卡
bash scripts/hybrid_search/run_hybrid_in1k_resnet18_train.sh 0 <json_file_path\> <GPU_number>

```

## 开发环境
work_dirs地址: 203服务器 /data/work_dirs/wyh
``` bash

conda activate wyh_graduate

```
安装新python包时，尽量使用pip

## 可以做的实验：
1. 学习率 step cosine
2. 优化器 SGD Adam
3. 模型内部结构
## 添加权重融合模块
Pytorch 官方实现的conv+bn权重融合
https://github.com/pytorch/pytorch/blob/master/torch/nn/utils/fusion.py
## 添加模型层剪枝模块

## 记录疑问与问题
### train/val代码写在哪个模块里？
*回答*：有可能Algorithm和Searcher两个都有。
- AutoSlim: 因为流程是：先训练，再搜索。所以模型训练代码写Algorithm里，Searcher有一些val的
- Hybrid： 因为流程是：先搜索，再训练。所以模型训练代码写在Algorithm里，Searcher有一些val的。.提取特征的代码写在了Searcher里。
- Fusion: 2.25流程是：先根据特征图/权重计算层重要性，所以是模型训练代码写在Algorithm里，提取特征的代码写在Algorithm里

### Question
``` shell
config.py:
    resume_from 9
    load_from
<main>.py:
    checkpoint
```
差别是什么？都是在什么时候加载的？会互相覆盖吗？
### BUG

Autoslim和Hybrid都报错：

```
super(RatioPruner, self).prepare_from_supernet(supernet)

self.trace_norm_conv_links(pseudo_loss.grad_fn, module2name,

File "/home/wyh/wuyuhang/mmrazor/mmrazor/models/pruners/structure_pruning.py", line 710, in trace_norm_conv_links
    conv_grad_fn = conv_grad_fn.next_functions[0][0]
```
循环到conv_grad_fn为None后，还在继续循环

``` shell
AttributeError: AutoSlim: 'NoneType' object has no attribute 'next_functions'
```
## 待办事项

1. mmseg config&checkpoint
2. mmcls resnet/mobilenetv2 checkpoint