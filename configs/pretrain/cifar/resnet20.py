_base_ = [
    '../../_base_/datasets/cifar10_bs16.py',
    '../../_base_/schedules/cifar10_bs128.py', 
    '../../_base_/default_runtime.py'
]
# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNet_CIFAR',
        depth=20,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=10,
        in_channels=64,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    ))
# learning policy
# lr_config = dict(
#     _delete_=True,
#     policy='CosineAnnealing',
#     min_lr=0,
#     warmup='linear',
#     warmup_iters=10,
#     warmup_ratio=0.5)
# lr_config = dict(policy='step', step=[100, 150,])
lr_config = dict(policy='step', step=[60, 120, 160], gamma=0.2)
runner=dict(type='EpochBasedRunner', max_epochs=200)
# data=dict(samples_per_gpu=4096, workers_per_gpu=8)
data=dict(samples_per_gpu=8192, workers_per_gpu=8)
auto_scale_lr = dict(base_batch_size=128)