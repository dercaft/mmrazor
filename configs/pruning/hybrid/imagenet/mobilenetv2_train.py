_base_=[
    './mobilenetv2_search.py'
]
model=dict()
checkpoint_config=dict(interval=10)

algorithm=dict(
    type='ChannelPruningAlgoritm',
    architecture=dict(type='MMClsArchitecture', model=model),
    distiller=None,
    pruner=dict(type='ChannelPruner'),
    retraining=True,
    # bn_training_mode=False,
    input_shape=(3,224,224),    
)
# optimizer
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
#
# learning policy
lr_config = dict(
    _delete_=True,
    policy='CosineAnnealing',
    min_lr=0,
    warmup='linear',
    warmup_iters=50,
    warmup_ratio=0.25)
# lr_config = dict(policy='step', step=[100, 150,])
# lr_config = dict(policy='step', step=[60, 120, 160], gamma=0.2)
runner=dict(type='EpochBasedRunner', max_epochs=200)
# use_ddp_wrapper=True
# data=dict(samples_per_gpu=96, workers_per_gpu=8)
# data=dict(samples_per_gpu=2048, workers_per_gpu=8)
evaluation=dict(interval=1, metric='accuracy', save_best='accuracy_top-1')