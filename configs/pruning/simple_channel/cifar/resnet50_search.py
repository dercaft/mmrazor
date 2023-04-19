_base_ = [
    # '../../../_base_/models/resnet50_cifar.py', 
    '../../../_base_/datasets/cifar10_bs16.py',
    '../../../_base_/schedules/cifar10_bs128.py', 
    '../../../_base_/default_runtime.py'
]
model = dict(
    type='mmcls.ImageClassifier',
    backbone=dict(
        type='ResNet_CIFAR',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=10,
        in_channels=2048,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    ))
algorithm=dict(
    type='ChannelPruningAlgoritm',
    # architecture=dict(type='MMClsArchitecture', model={{_base_.model}}),
    architecture=dict(type='MMClsArchitecture', model=model),
    distiller=None,
    pruner=dict(type='ChannelPruner',pruning_metric_name='L2'),
    retraining=False,
    input_shape=(3,32,32),
)
runner=dict(type='EpochBasedRunner',max_epochs=1)
# use_ddp_wrapper=True
data=dict(samples_per_gpu=128, workers_per_gpu=4)