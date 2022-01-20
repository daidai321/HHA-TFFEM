_base_ = [
    '../_base_/models/rgbd/retinanet_r50_fpn.py',
    '../_base_/datasets/kitti_retina_detection.py',
    '../_base_/schedules/schedule_2x.py',
    '../_base_/default_runtime.py'
]
# optimizer
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    # [7] yields higher performance than [6]
    step=[14])
runner = dict(
    type='EpochBasedRunner', max_epochs=16)  # actual epoch = 8 * 8 = 64

# load_from = 'D:/paper_segmentation/code/mmdetection/checkpoints/retinanet.pth'
