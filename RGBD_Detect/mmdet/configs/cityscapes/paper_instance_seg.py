_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py',
    '../_base_/default_runtime.py'
]

dataset_type = 'CityscapesDataset'
data_root = 'D:/dataset/RGB/cityspace/'

# img_norm_cfg = dict(
#     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

img_norm_cfg = dict(mean=[93.83309082, 98.76049672, 95.87739305], std=[78.78147396, 80.1303281, 81.19954495],
                    to_rgb=True, dmean=[100.63431417615382, 76.80792407431142, 47.81952119941237],
                    dstd=[50.808564486310466, 87.81012534457817, 44.61858198193684])

train_pipeline = [
    dict(type='LoadRGBDImageFromFile2'),
    dict(type='LoadRGBDAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='Resize', img_scale=[(1600, 704), (1600, 800)], keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Pad', size_divisor=32),
    dict(type='NormalizeRGBD', **img_norm_cfg),
    dict(type='DefaultFormatBundleRGBD'),
    dict(type='Collect', keys=['img', 'imgd', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]

test_pipeline = [
    dict(type='LoadRGBDImageFromFile2'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1600, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Pad', size_divisor=32),
            dict(type='NormalizeRGBD', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img', 'imgd']),
            dict(type='Collect', keys=['img', 'imgd']),
        ])
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type='RepeatDataset',
        times=4,
        dataset=dict(
            type=dataset_type,
            ann_file=data_root +
                     'annotations/instancesonly_filtered_gtFine_train.json',
            img_prefix=data_root + 'images/',
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        ann_file=data_root +
                 'annotations/instancesonly_filtered_gtFine_val.json',
        img_prefix=data_root + 'images/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root +
                 'annotations/instancesonly_filtered_gtFine_test.json',
        img_prefix=data_root + 'images/',
        pipeline=test_pipeline))
evaluation = dict(metric=['bbox', 'segm'])

model = dict(
    backbone=dict(
        # type='FusionNet',
        type='fusionnet_CBAM_RGBD',
    ),
    roi_head=dict(
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=8,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
        mask_head=dict(
            type='FCNMaskHead',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=8,
            loss_mask=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))))
# optimizer
# lr is set for a batch size of 8
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    # [7] yields higher performance than [6]
    step=[7])
runner = dict(
    type='EpochBasedRunner', max_epochs=8)  # actual epoch = 8 * 8 = 64
log_config = dict(interval=100)
# For better, more stable performance initialize from COCO
load_from = 'D:\paper_segmentation\code\mmdetection\checkpoints\maskrcnn.pth'
