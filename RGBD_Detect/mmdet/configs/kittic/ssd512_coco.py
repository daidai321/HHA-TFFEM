_base_ = 'ssd300_coco.py'
input_size = 512
model = dict(
    neck=dict(
        out_channels=(512, 1024, 512, 256, 256, 256, 256),
        level_strides=(2, 2, 2, 2, 1),
        level_paddings=(1, 1, 1, 1, 1),
        last_kernel_size=4),
    bbox_head=dict(
        in_channels=(512, 1024, 512, 256, 256, 256, 256),
        anchor_generator=dict(
            type='SSDAnchorGenerator',
            scale_major=False,
            input_size=input_size,
            basesize_ratio_range=(0.1, 0.9),
            strides=[8, 16, 32, 64, 128, 256, 512],
            ratios=[[2], [2, 3], [2, 3], [2, 3], [2, 3], [2], [2]])))
# dataset settings
dataset_type = 'KITTIDataset'
data_root = 'D:/dataset/RGB-D/kitti_split/splitimg/'
# img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[1, 1, 1], to_rgb=True)
img_norm_cfg = dict(mean=[93.83309082, 98.76049672, 95.87739305], std=[1, 1, 1],
                    to_rgb=True, dmean=[208.63431417615382, 76.80792407431142, 47.81952119941237],
                    dstd=[1, 1, 1])
CLASSES = ('Pedestrian',)
train_pipeline = [
    dict(type='LoadRGBDImageFromFile', to_float32=True),
    dict(type='LoadRGBDAnnotations', with_bbox=True),
    dict(
        type='PhotoMetricDistortionRGBD',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(
        type='ExpandRGBD',
        ratio_range=(1, 4)),
    dict(
        type='MinIoURandomCropRGBD',
        min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
        min_crop_size=0.3),
    dict(type='Resize', img_scale=(512, 512), keep_ratio=False),
    dict(type='NormalizeRGBD', **img_norm_cfg),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='DefaultFormatBundleRGBD'),
    dict(type='Collect', keys=['img', 'imgd', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadRGBDImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='NormalizeRGBD', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img', 'imgd']),
            dict(type='Collect', keys=['img', 'imgd']),
        ])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=2,
    train=dict(
        _delete_=True,
        type='RepeatDataset',
        times=2,
        dataset=dict(
            type=dataset_type,
            ann_file=data_root + 'train1.json',
            img_prefix=data_root,
            classes=CLASSES,
            pipeline=train_pipeline)),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))
# optimizer
optimizer = dict(type='SGD', lr=2e-3, momentum=0.9, weight_decay=5e-4)
optimizer_config = dict(_delete_=True)
custom_hooks = [
    dict(type='NumClassCheckHook'),
    dict(type='CheckInvalidLossHook', interval=50, priority='VERY_LOW')
]

load_from = 'D:/paper1/code/python/mmdetectionFCOS/checkpoints/ssd500.pth'
