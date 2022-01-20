_base_ = 'fcos_r50_caffe_fpn_gn-head_1x_coco.py'

model = dict(
    bbox_head=dict(
        norm_on_bbox=True,
        centerness_on_reg=True,
        dcn_on_last_conv=True,
        center_sampling=True,
        conv_bias=True,
        loss_bbox=dict(type='GIoULoss', loss_weight=1.0)),
    # training and testing settings
    test_cfg=dict(nms=dict(type='nms', iou_threshold=0.6)))

# dataset settings
img_norm_cfg = dict(mean=[93.83309082, 98.76049672, 95.87739305], std=[1.0, 1.0, 1.0],
                    to_rgb=True, dmean=[208.63431417615382, 76.80792407431142, 47.81952119941237],
                    dstd=[1.0, 1.0, 1.0])
train_pipeline = [
    dict(type='LoadRGBDImageFromFile'),
    dict(type='LoadRGBDAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='NormalizeRGBD', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'imgd', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadRGBDImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='NormalizeRGBD', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img', 'imgd']),
            dict(type='Collect', keys=['img', 'imgd']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))
optimizer_config = dict(_delete_=True, grad_clip=None)

lr_config = dict(warmup='linear')

load_from = 'D:/paper_segmentation/code/mmdetection/checkpoints/fcos.pth'