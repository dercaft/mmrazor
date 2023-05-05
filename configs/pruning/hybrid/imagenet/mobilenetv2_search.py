_base_ = [
    # '../../../_base_/models/mobilenet_v2_1x.py',
    '../../../_base_/datasets/imagenet_bs64_pil_resize.py',
    './imagenet_bs256_150e_coslr_warmup.py',
    '../../../_base_/default_runtime.py'
]

model = dict(
    type='mmcls.ImageClassifier',
    backbone=dict(type='MobileNetV2', widen_factor=1.0),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=1280,
        loss=dict(
            type='LabelSmoothLoss',
            mode='original',
            label_smooth_val=0.1,
            loss_weight=1.0),
        topk=(1, 5),
    ))

algorithm = dict(
    type='ChannelPruningAlgoritm',
    architecture=dict(type='MMClsArchitecture', model=model),
    distiller=None,
    pruner=dict(
        type='ChannelPruner',
        ),
    retraining=False,
    # bn_training_mode=True,
    input_shape=(3,224,224))

runner = dict(type='EpochBasedRunner', max_epochs=1)

use_ddp_wrapper = True

# use_ddp_wrapper=True
searcher=dict(
    type='CKAEvolutionSearcher',
    reduction_ratio=0.5,
    metrics='accuracy',
    candidate_pool_size=100,
    max_epoch=20,
    rand_seed=0,
    metric='FPGM',
    metric_options=[],
)
data=dict(
    samples_per_gpu=100, 
    workers_per_gpu=8,
    train=dict(
        data_prefix='/data/imagenet/ILSVRC2012_img_train',
        ),
    val=dict(
        data_prefix='/data/imagenet/val',
        ann_file=None,
        ),
    test=dict(
        # replace `data/val` with `data/test` for standard test
        data_prefix='/data/imagenet/val',
        ann_file=None,
        )
    )