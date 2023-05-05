
_base_=[
    './resnet18_in1k_search.py'
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
# ######
# # optimizer
# optim_wrapper = dict(
#     optimizer=dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001))

# # learning policy
# param_scheduler = dict(
#     type='MultiStepLR', by_epoch=True, milestones=[30, 60, 90], gamma=0.1)

# # train, val, test setting
# train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=1)
# val_cfg = dict()
# test_cfg = dict()

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# based on the actual training batch size.
# auto_scale_lr = dict(base_batch_size=256)