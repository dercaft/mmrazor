_base_ = [
    '../../../_base_/datasets/mmcls/imagenet_bs256_autoslim.py',
    '../../../_base_/schedules/mmcls/imagenet_bs2048_autoslim.py',
    '../../../_base_/mmcls_runtime.py'
]

model = dict(
    type='mmcls.ImageClassifier',
    backbone=dict(type='MobileNetV2', widen_factor=1.5),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=1920,
        loss=dict(
            type='LabelSmoothLoss',
            mode='original',
            label_smooth_val=0.1,
            loss_weight=1.0),
        topk=(1, 5),
    ))

algorithm = dict(
    type='CommonPruningAlgorithm',
    architecture=dict(type='MMClsArchitecture', model=model),
    distiller=None,
    pruner=dict(
        type='CommonPruner',
        ),
    retraining=False,
    # bn_training_mode=True,
    input_shape=(3,224,224))

runner = dict(type='EpochBasedRunner', max_epochs=1)

use_ddp_wrapper = True

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
data=dict(samples_per_gpu=64, workers_per_gpu=4)
