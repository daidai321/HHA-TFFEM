_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/kittirgbdet.py',
    '../_base_/schedules/schedule_2x.py',
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
    step=[7])
runner = dict(
    type='EpochBasedRunner', max_epochs=8)  # actual epoch = 8 * 8 = 64

load_from = 'D:/paper_segmentation/code/mmdetection/checkpoints/fasterrcnn2x.pth'

# load_from = 'D:/paper_segmentation/code/mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
