# dataset settings
dataset_type = 'KITTIDataset'
# data_root = 'D:/dataset/RGB-D/KITTI/training/'
# data_root_train = 'D:/dataset/RGB-D/cityspace_out/'

data_root = 'D:/dataset/RGB-D/kitti_out/'

CLASSES = ('Pedestrian',)

# img_norm_cfg = dict(
#     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)


# img_norm_cfg = dict(mean=[93.83309082, 98.76049672, 95.87739305], std=[78.78147396, 80.1303281, 81.19954495],
#                     to_rgb=True, dmean=[11.218104950173572, 65.5191944010483, 173.98679371843122],
#                     dstd=[32.83914604212737, 69.79726543682794, 35.16480852755739])

# dhh
# img_norm_cfg = dict(mean=[93.83309082, 98.76049672, 95.87739305], std=[78.78147396, 80.1303281, 81.19954495],
#                     to_rgb=True, dmean=[30.47737095, 132.36374398, 89.85209354],
#                     dstd=[46.62792831, 100.31391884, 74.31366146])

# norm
# img_norm_cfg = dict(mean=[93.83309082, 98.76049672, 95.87739305], std=[78.78147396, 80.1303281, 81.19954495],
#                     to_rgb=True, dmean=[77.85651739, 24.59565985, 82.77221306],
#                     dstd=[66.97329861, 39.99295247, 77.30677607])

img_norm_cfg = dict(mean=[93.83309082, 98.76049672, 95.87739305], std=[78.78147396, 80.1303281, 81.19954495],
                    to_rgb=True, dmean=[35.5591357039706, 152.47645700172026, 72.51905036127917],
                    dstd=[38.6769951359651, 31.428243265866456, 67.12998721226808])

# img_norm_cfg = dict(mean=[93.83309082, 98.76049672, 95.87739305], std=[78.78147396, 80.1303281, 81.19954495],
#                     to_rgb=True, dmean=[30], dstd=[20])

# img_norm_cfg = dict(
#     mean=[.0, .0, .0], std=[255.0, 255.0, 255.0], to_rgb=True)

# img_norm_cfg = dict(mean=[0.0, 0.0, 0.0], std=[255.0, 255.0, 255.0],
#                     to_rgb=True, dmean=[0.0, 0.0, 0.0], dstd=[255.0, 255.0, 255.0])

train_pipeline = [
    dict(type='LoadRGBDImageFromFile', to_float32=True),
    dict(type='LoadRGBDAnnotations', with_bbox=True, with_bbox3d=True),
    # dict(
    #     type='PhotoMetricDistortionRGBD',
    #     brightness_delta=32,
    #     contrast_range=(0.5, 1.5),
    #     saturation_range=(0.5, 1.5),
    #     hue_delta=18),

    dict(
        type='ExpandRGBD',
        ratio_range=(1, 1.5)),
    dict(
        type='MinIoURandomCropRGBD',
        min_ious=(0.6, 0.8),
        min_crop_size=0.6),

    dict(type='Resize', img_scale=[(1333, 416), (1333, 672)], keep_ratio=False),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='NormalizeRGBD', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundleRGBD'),
    dict(type='Collect', keys=['img', 'imgd', 'gt_bboxes', 'gt_labels', 'gt_dims', 'gt_locs', 'gt_rots']),
]
test_pipeline = [
    dict(type='LoadRGBDImageFromFile', to_float32=True),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 416),
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
evaluation = dict(interval=1, metric='bbox')
