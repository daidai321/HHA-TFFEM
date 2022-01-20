# dataset settings
dataset_type = 'KITTIDataset'
data_root = 'D:/dataset/RGB-D/KITTI/training/'
CLASSES = ('Pedestrian',)

# img_norm_cfg = dict(
#     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# img_norm_cfg = dict(mean=[93.83309082, 98.76049672, 95.87739305], std=[78.78147396, 80.1303281, 81.19954495],
#                     to_rgb=True, dmean=[0.86, 3.16, 132.0], dstd=[12.35, 23.17, 22.8])

# img_norm_cfg = dict(mean=[93.83309082, 98.76049672, 95.87739305], std=[78.78147396, 80.1303281, 81.19954495],
#                     to_rgb=True, dmean=[54.5, 54.5, 54.5], dstd=[38.85, 38.85, 38.85])

# img_norm_cfg = dict(mean=[93.83309082, 98.76049672, 95.87739305], std=[78.78147396, 80.1303281, 81.19954495],
#                     to_rgb=True, dmean=[30], dstd=[20])

# img_norm_cfg = dict(
#     mean=[.0, .0, .0], std=[255.0, 255.0, 255.0], to_rgb=True)

img_norm_cfg = dict(mean=[0.0, 0.0, 0.0], std=[255.0, 255.0, 255.0],
                   to_rgb=True, dmean=[0.0, 0.0, 0.0], dstd=[255.0, 255.0, 255.0])

train_pipeline = [
    dict(type='LoadRGBDImageFromFile', to_float32=True),
    dict(type='LoadRGBDAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1216, 352), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='NormalizeRGBD', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundleRGBD'),
    dict(type='Collect', keys=['img', 'imgd', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadRGBDImageFromFile', to_float32=True),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1216, 352),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='NormalizeRGBD', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img', 'imgd']),
            dict(type='Collect', keys=['img', 'imgd']),
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        classes=CLASSES,
        ann_file=data_root + 'train.json',
        img_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        classes=CLASSES,
        ann_file=data_root + 'val.json',
        img_prefix=data_root,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=CLASSES,
        ann_file=data_root + 'val.json',
        img_prefix=data_root,
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')
