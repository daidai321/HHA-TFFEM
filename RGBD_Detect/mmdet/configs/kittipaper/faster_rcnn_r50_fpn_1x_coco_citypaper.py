_base_ = [
    '../_base_/models/rgbd/faster_rcnn_r50_fpn_kitti_paper.py',
    # '../_base_/models/rgbd/cascade_rcnn_r50_fpn_kitti_paper.py',
    '../_base_/datasets/rgbd/kitti_detection_paper.py',
    '../_base_/default_runtime.py'
]

dataset_type = 'KITTIDataset'
data_root = 'D:/dataset/RGB-D/kitti_out/'
data_root_test = 'D:/dataset/RGB-D/KITTI/testing/outresult/testsample/'
CLASSES = ('Pedestrian',)
img_norm_cfg = dict(mean=[93.83309082, 98.76049672, 95.87739305], std=[78.78147396, 80.1303281, 81.19954495],
                    to_rgb=True, dmean=[100.63431417615382, 76.80792407431142, 47.81952119941237],
                    dstd=[50.808564486310466, 87.81012534457817, 44.61858198193684])

train_pipeline = [
    dict(type='LoadRGBDImageFromFile', to_float32=True),
    dict(type='LoadRGBDAnnotations', with_bbox=True),

    dict(type='ExpandRGBD', ratio_range=(1, 1.5)),
    dict(type='MinIoURandomCropRGBD', min_ious=(0.6, 0.8), min_crop_size=0.6),

    # dict(
    #     type='RandomAffineRGBD',
    #     scaling_ratio_range=(0.8, 1.2),
    #     border=(-16, -48)),
    # dict(type='Resize', img_scale=[(1344, 512), (1120, 384)], keep_ratio=True),
    dict(type='Resize', img_scale=[(1912, 768), (1720, 672)], keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Pad', size_divisor=32),

    # dict(
    #     type='PhotoMetricDistortionRGBD',
    #     brightness_delta=32,
    #     contrast_range=(0.5, 1.5),
    #     saturation_range=(0.5, 1.5),
    #     hue_delta=9),

    dict(type='NormalizeRGBD', **img_norm_cfg),
    dict(type='DefaultFormatBundleRGBD'),
    dict(type='Collect', keys=['img', 'imgd', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadRGBDImageFromFile2', to_float32=True),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1912, 768),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
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
        times=1,
        dataset=dict(
            type=dataset_type,
            classes=CLASSES,
            ann_file=data_root + 'trainallcity.json',
            img_prefix=data_root,
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        classes=CLASSES,
        ann_file=data_root_test + 'valtest.json',
        img_prefix=data_root_test,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=CLASSES,
        ann_file=data_root_test + 'valtest.json',
        img_prefix=data_root_test,
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    # [7] yields higher performance than [6]
    step=[12, 1])
runner = dict(
    type='EpochBasedRunner', max_epochs=16)  # actual epoch = 8 * 8 = 64

# load_from = 'D:/paper_segmentation/code/mmdetection/checkpoints/fasterrcnn2x.pth'
# load_from = 'D:/paper_segmentation/code/mmdetection/checkpoints/799fusionrgbd.pth'
# load_from = 'D:/paper_segmentation/code/mmdetection/tools/work_dirs/faster_rcnn_r50_fpn_1x_coco_testup/epoch_16.pth'
load_from = 'D:/paper_segmentation/code/mmdetection/tools/work_dirs/faster_rcnn_r50_fpn_1x_coco_paper/831.pth'
# load_from = 'D:/paper_segmentation/code/mmdetection/checkpoints/cascade.pth'2
