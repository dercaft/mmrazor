from platform import architecture


_base_=[
    './resnet18_cifar10_search.py'
]
model=dict()
checkpoint_config=dict(interval=100)

algorithm=dict(
    architecture=dict(type='MMClsArchitecture',model=model),
    retraining=True,
    bn_training_mode=False,
    input_shape=(3,32,32),    
)
runner=dict(type='EpochBasedRunner', max_epochs=600)
# use_ddp_wrapper=True
data=dict(samples_per_gpu=128, workers_per_gpu=4)
evaluation=dict(interval=1, metric='accuracy', save_best='accuracy_top-1')