# dataset settings
dataset_type = 'KITTIDataset'
# data_root = 'D:/dataset/RGB-D/KITTI/training/'
# data_root_train = 'D:/dataset/RGB-D/cityspace_out/'

# data_root = 'D:/dataset/RGB-D/kitti_out/'
# data_root = 'D:/dataset/RGB-D/epfl_out/'
data_root = 'D:/dataset/RGB-D/kitti_split/splitimg/'

CLASSES = ('Pedestrian',)

# img_norm_cfg = dict(
#     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# img_norm_cfg = dict(mean=[93.83309082, 98.76049672, 95.87739305], std=[78.78147396, 80.1303281, 81.19954495],
#                     to_rgb=True, dmean=[0.86, 3.16, 132.0], dstd=[12.35, 23.17, 22.8])

# img_norm_cfg = dict(mean=[93.83309082, 98.76049672, 95.87739305], std=[78.78147396, 80.1303281, 81.19954495],
#                     to_rgb=True, dmean=[35.5591357039706, 152.47645700172026, 72.51905036127917],
#                     dstd=[38.6769951359651, 31.428243265866456, 67.12998721226808])

img_norm_cfg = dict(mean=[128.0, 128.0, 128.0], std=[1.0, 1.0, 1.0],
                    to_rgb=True, dmean=[128.0, 128.0, 128.0],
                    dstd=[1.0, 1.0, 1.0])

# img_norm_cfg = dict(mean=[93.83309082, 98.76049672, 95.87739305], std=[78.78147396, 80.1303281, 81.19954495],
#                     to_rgb=True, dmean=[30], dstd=[20])

# img_norm_cfg = dict(
#     mean=[.0, .0, .0], std=[255.0, 255.0, 255.0], to_rgb=True)

# img_norm_cfg = dict(mean=[0.0, 0.0, 0.0], std=[255.0, 255.0, 255.0],
#                     to_rgb=True, dmean=[0.0, 0.0, 0.0], dstd=[255.0, 255.0, 255.0])

train_pipeline = [
    dict(type='LoadRGBDImageFromFile', to_float32=True),
    dict(type='LoadRGBDAnnotations', with_bbox=True),
    # dict(
    #     type='PhotoMetricDistortionRGBD',
    #     brightness_delta=32,
    #     contrast_range=(0.5, 1.5),
    #     saturation_range=(0.5, 1.5),
    #     hue_delta=18),

    dict(
        type='ExpandRGBD',
        ratio_range=(1, 1.2)),
    dict(
        type='MinIoURandomCropRGBD',
        min_ious=(0.8, 0.9),
        min_crop_size=0.9),

    dict(type='Resize', img_scale=[(832, 640)], keep_ratio=True),

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
        img_scale=(832, 640),
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
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='RepeatDataset',
        times=2,
        dataset=dict(
            type=dataset_type,
            classes=CLASSES,
            ann_file=data_root + 'train.json',
            img_prefix=data_root,
            pipeline=train_pipeline)),
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
evaluation = dict(interval=2, metric='bbox')
