_base_=[
 '../../../_base_/datasets/cifar10_bs16.py',
    '../../../_base_/schedules/cifar10_bs128.py',
    '../../../_base_/default_runtime.py']

model=dict(
    type='mmcls.ImageClassifier',
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
    )
)
algorithm=dict(
    type='ChannelPruningAlgoritm',
    architecture=dict(type='MMClsArchitecture', model=model),
    distiller=None,
    pruner=dict(type='ChannelPruner'),
    retraining=False,
    input_shape=(3,32,32),
)
# runner=dict(type='EpochBasedRunner',max_epochs=1)
# use_ddp_wrapper=True
searcher=dict(
    type='CKAEvolutionSearcher',
    reduction_ratio=0.5,
    metrics='accuracy',
    candidate_pool_size=100,
    max_epoch=10,
    rand_seed=0,
    metric='CDP',
    metric_options=["L1","FPGM","HRANK","APOZ"],
)
data=dict(samples_per_gpu=128, workers_per_gpu=8)
lr_config = dict(
    _delete_=True,
    policy='CosineAnnealing',
    min_lr=0,
)
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# lr_config = dict(policy='step', step=[100, 150,])
# lr_config = dict(policy='step', step=[60, 120, 160], gamma=0.2)
runner=dict(type='EpochBasedRunner', max_epochs=600)
# use_ddp_wrapper=True
# data=dict(samples_per_gpu=128, workers_per_gpu=4)
# data=dict(samples_per_gpu=2048, workers_per_gpu=8)
evaluation=dict(interval=1, metric='accuracy', save_best='accuracy_top-1')