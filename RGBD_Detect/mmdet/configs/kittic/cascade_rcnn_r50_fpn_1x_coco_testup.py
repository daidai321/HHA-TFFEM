_base_ = [
    '../_base_/models/cascade_rcnn_r50_fpn_kitti_testup_param.py',
    '../_base_/datasets/kitti_detection_testup_split_test.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    # [7] yields higher performance than [6]
    step=[10])
runner = dict(
    type='EpochBasedRunner', max_epochs=12)  # actual epoch = 8 * 8 = 64

load_from = 'D:/paper_segmentation/code/mmdetection/checkpoints/cascade.pth'
# load_from = 'D:/paper_segmentation/code/mmdetection/checkpoints/799fusionrgbd.pth'
