
_base_=[
    './resnet18_cifar10_search.py'
]
model=dict()
checkpoint_config=dict(interval=100)

algorithm=dict(
    type='ChannelPruningAlgoritm',
    architecture=dict(type='MMClsArchitecture', model=model),
    distiller=None,
    pruner=dict(type='ChannelPruner'),
    retraining=True,
    # bn_training_mode=False,
    input_shape=(3,32,32),    
)
# optimizer
# optimizer = dict(type='SGD', lr=0.5, momentum=0.9, weight_decay=0.0001)
# optimizer_config = dict(grad_clip=None)
#
# learning policy
# lr_config = dict(
#     _delete_=True,
#     policy='CosineAnnealing',
#     min_lr=0,
#     warmup='linear',
#     warmup_iters=10,
#     warmup_ratio=0.25)
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