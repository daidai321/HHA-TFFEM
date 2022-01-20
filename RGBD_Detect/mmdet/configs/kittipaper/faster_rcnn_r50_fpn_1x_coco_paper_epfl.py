_base_ = [
    '../_base_/models/rgbd/faster_rcnn_r50_fpn_kitti_paper.py',
    '../_base_/datasets/rgbd/kitti_detection_paper_epfl.py',
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
    step=[12, 14])
runner = dict(
    type='EpochBasedRunner', max_epochs=16)  # actual epoch = 8 * 8 = 64

load_from = 'D:/paper_segmentation/code/mmdetection/checkpoints/fasterrcnn2x.pth'
# load_from = 'D:/paper_segmentation/code/mmdetection/checkpoints/799fusionrgbd.pth'
# load_from = 'D:/paper_segmentation/code/mmdetection/tools/work_dirs/faster_rcnn_r50_fpn_1x_coco_testup/epoch_16.pth'
