_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn_kitti.py',
    '../_base_/datasets/kitti_detection.py',
    '../_base_/default_runtime.py'
]

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.0001,
    # [7] yields higher performance than [6]
    step=[14])
runner = dict(
    type='EpochBasedRunner', max_epochs=16)  # actual epoch = 8 * 8 = 64

load_from = 'D:/paper_segmentation/code/mmdetection/checkpoints/fasterrcnn2x.pth'
# load_from = 'D:/paper_segmentation/code/mmdetection/checkpoints/fasterfusion.pth'