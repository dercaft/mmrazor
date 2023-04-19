_base_=[
 '../../../_base_/datasets/cifar10_bs16.py',
    '../../../_base_/schedules/cifar10_bs128.py',
    '../../../_base_/default_runtime.py']
model=dict(
    type='mmcls.ImageClassifier',
    backbone=dict(
        type='ResNet_CIFAR',
        depth=18,
        num_stages=4,
        out_indices=(3,),
        style='pytorch',
    ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=10,
        in_channels=512,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    )
)
algorithm=dict(
    type='ChannelMergingAlgoritm',
    architecture=dict(type='MMClsArchitecture', model=model),
    distiller=None,
    pruner=dict(type='ChannelPruner'),
    retraining=False,
    input_shape=(3,32,32),
)
runner=dict(type='EpochBasedRunner',max_epochs=1)
# use_ddp_wrapper=True
searcher=dict(
    type='CKAEvolutionSearcher',
    reduction_ratio=0.9,
    metrics='accuracy',
    candidate_pool_size=50,
    max_epoch=20,
    rand_seed=0,
    metric='FPGM',
    metric_options=[],
)
data=dict(samples_per_gpu=1024, workers_per_gpu=4)