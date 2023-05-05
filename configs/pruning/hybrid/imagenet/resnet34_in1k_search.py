_base_=[
    '../../../_base_/datasets/imagenet_bs32.py',
    '../../../_base_/schedules/imagenet_bs256.py',
    '../../../_base_/default_runtime.py'
]

# model settings
model = dict(
    type='mmcls.ImageClassifier',
    backbone=dict(
        type='ResNet',
        depth=34,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=512,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))
algorithm=dict(
    type='ChannelPruningAlgoritm',
    architecture=dict(type='MMClsArchitecture', model=model),
    distiller=None,
    pruner=dict(type='ChannelPruner'),
    retraining=False,
    input_shape=(3,224,224),
)
runner=dict(type='EpochBasedRunner',max_epochs=1)
# use_ddp_wrapper=True
searcher=dict(
    type='CKAEvolutionSearcher',
    reduction_ratio=0.5,
    metrics='accuracy',
    candidate_pool_size=100,
    max_epoch=20,
    rand_seed=0,
    metric='CDP',
    metric_options=["L1","FPGM","HRANK","APOZ"],
)
data=dict(
    workers_per_gpu=16,
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
