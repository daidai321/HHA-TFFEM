# HHA-TFFEM
code for the paper "3D Sensor-based Pedestrian Detection by Integrating Improved HHA Encoding and Two-branch Feature Fusion."

The code consists of two parts, one is the improved HHA encoding (Improved HHA) , and the other is RGB-D pedestrian detection (RGBD_Detect).


## Improved HHA
1. the code is implemented in the Matlab platform and win10.

2.  depth image and camera intrinsics are used as input.

3. The algorithm is based on the work [**"Learning Rich Features from RGB-D Images for Object Detection and Segmentation"**](https://link.springer.com/chapter/10.1007/978-3-319-10584-0_23), and the code also refers to the [**official code**](https://github.com/s-gupta/rcnn-depth) provided by them.

4. The improved HHA encoding is faster and the encoding results are more consistent. We validate the performance of our HHA encoding method on several RGB-D datasets, including [**KITTI**](http://www.cvlibs.net/datasets/kitti/index.php), [**EPFL**](https://www.epfl.ch/labs/cvlab/data/data-rgbd-pedestrian/), [**KTP**](http://www.dei.unipd.it/~munaro/KTP-dataset.html), and [**UNIHall**](http://www2.informatik.uni-freiburg.de/~spinello/RGBD-dataset.html). Detailed comparison results can be found in the paper.

## RGBD Pedestrain Detection

We proposed a two-branch feature fusion extraction module (TFFEM) to obtain both modalities' local and global features. Based on TFFEM, an RGB-D pedestrian detection network is designed to locate the people, with RGB and HHA images as inputs. 

### Install dependence

the code is based on the [**mmdetection**](https://github.com/open-mmlab/mmdetection). Therefore, the mmdetection is needed to install according to their guidelines. Then replace the mmdet file we provided.

### Prepare the dataset

Take the KITTI dataset as an example. First download the dataset and create some directories. 
```plain
└── KITTI_DATASET_ROOT
       ├── training    <-- 7481 train data
       |   ├── images <-- (image_2)
       |   ├── hha
       |   ├── labels <-- (label_2)
       └── testing     <-- 7580 test data
           ├── images <-- (image_2)
           ├── hha
           ├── labels <-- (label_2)
```
The HHA data obtained from the previous section or you can directly download from [**here**](https://pan.baidu.com/s/1IH6HOAMwgIBd7t617FkLcg) (Extraction code：TFFE)


Then create the JSON file and replace it in the mmdetection project.

```
python createjson.py
```


### train

```
python train.py
```




