_base_ = [
    '../../_base_/datasets/cifar10_bs16.py',
    '../../_base_/schedules/cifar10_bs128.py', 
    '../../_base_/default_runtime.py'
]
# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(type='VGG', depth=11, num_classes=10),
    neck=None,
    head=dict(
        type='ClsHead',
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))