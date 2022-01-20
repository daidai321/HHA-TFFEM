import os
import numpy as np
import cv2
from tqdm import tqdm
import json
from datapremmdetection.util.utilmain import get_label_anno


def createtrainvaltxt(imgdir, trainrate=0.5):
    allimgs = os.listdir(imgdir)

    imgcity = []
    imglisttrain = []
    imglistval = []
    allkitti = []

    for imgname in allimgs:
        if 'leftImg8bit' in imgname:
            imgcity.append(imgname)
        else:
            allkitti.append(imgname)

    randomlist = np.random.permutation(len(allkitti))

    trainthres = trainrate * len(allkitti)
    for listidex in range(len(randomlist)):
        if listidex < trainthres:
            imglisttrain.append(int(allkitti[randomlist[listidex]][:-4]))
        else:
            imglistval.append(int(allkitti[randomlist[listidex]][:-4]))

    imglisttrain.sort()
    imglistval.sort()

    with open('train.txt', 'w') as f:
        for imgid in imglisttrain:
            f.writelines(str(imgid).zfill(6) + '.png\n')
        for imgname in imgcity:
            f.writelines(imgname + '\n')

    with open('val.txt', 'w') as f:
        for imgid in imglistval:
            f.writelines(str(imgid).zfill(6) + '.png\n')


means = [0.7189651278206727, 1.8049838344262712, 17.433151326053043]


# means = [0.35, 1.0, 17.433151326053043]


def main():
    basedir = 'D:/dataset/RGB-D/kitti_out/'
    labeldir = os.path.join(basedir, 'labels')
    imgdir = os.path.join(basedir, 'images')

    imgnames = os.listdir(imgdir)

    # cates = [{
    #     "id": 0,
    #     "name": "Pedestrian"
    # }]

    cates = [{
        "id": 0,
        "name": "Pedestrian"
    }, {
        "id": 1,
        "name": "Cyclist"
    }]

    # createtrainvaltxt(imgdir, 0.5)

    datatypes = ['train', 'val']
    for datatype in datatypes:

        annall = {}
        idimg = -1
        idann = -1
        annotate = []
        imgs = []
        imgalllist = []

        with open(datatype + '.txt', 'r') as f:
            imgalllist = [a.strip() for a in f.readlines()]

        for name in tqdm(imgnames):
            if name not in imgalllist:
                continue

            ishaveperson = False
            labelpath = os.path.join(labeldir, name[:-4] + '.txt')
            if not os.path.exists(labelpath):
                continue

            imgpath = os.path.join(imgdir, name)[:-4] + '.png'
            img = cv2.imread(imgpath, cv2.IMREAD_UNCHANGED)
            h, w = img.shape[:2]

            idimg = idimg + 1
            imgtmp = {
                'file_name': 'images/' + name,
                'id': idimg,
                'height': h,
                'width': w
            }
            imgs.append(imgtmp)

            if 'leftImg8bit' in name:

                if os.path.getsize(labelpath) == 0:
                    continue

                labels = np.loadtxt(labelpath).reshape(-1, 5)
                ishaveperson = True
                if not ishaveperson:
                    continue
                for label in labels:
                    idann = idann + 1
                    left = label[0]
                    top = label[1]
                    ws = label[2]
                    hs = label[3]
                    anntmp = {
                        'area': ws * hs,
                        'image_id': idimg,
                        'ignore': 0,
                        'iscrowd': 0,
                        'id': idann,
                        'bbox': [left, top, ws, hs],
                        # 'info3d': [-1, -1, -1, -1, -1, -1, -1],
                        # 'W': -100,
                        # 'H': -100,
                        # 'D': -100,
                        'category_id': int(label[4]),
                        'truncated': 0,
                        'occluded': 0,
                        'depth': 10.0
                    }
                    # anns['annotations'].append(anntmp)
                    annotate.append(anntmp)

            else:
                with open(labelpath, 'r') as f:
                    anns = [a.strip() for a in f.readlines()]
                for ann in anns:
                    if 'Pedestrian' in ann or 'Cyclist' in ann:
                        ishaveperson = True
                        break
                if not ishaveperson:
                    continue

                # camerapath = labelpath.replace('labels', 'camera')
                # camerainfo = np.loadtxt(camerapath)

                # fx = camerainfo[0, 0]
                # fy = camerainfo[0, 0]

                annos = get_label_anno(labelpath)
                for annidex in range(len(annos['name'])):
                    annname = annos['name'][annidex]
                    if annname == 'Pedestrian' or annname == 'Cyclist':
                        idann = idann + 1
                        # x1 y1 w h
                        x1 = annos['bbox'][annidex, 0]
                        y1 = annos['bbox'][annidex, 1]
                        annw = annos['bbox'][annidex, 2] - annos['bbox'][annidex, 0]
                        annh = annos['bbox'][annidex, 3] - annos['bbox'][annidex, 1]

                        # info3d = [annos['location'][annidex, 0], annos['location'][annidex, 1],
                        #           annos['location'][annidex, 2],
                        #           annos['dimensions'][annidex, 0], annos['dimensions'][annidex, 1],
                        #           annos['dimensions'][annidex, 2], annos['rotation_y'][annidex]]

                        catnumid = 0
                        if annname == 'Cyclist':
                            catnumid = 1

                        anntmp = {
                            'area': annw * annh,
                            'image_id': idimg,
                            'ignore': 0,
                            'iscrowd': 0,
                            'id': idann,
                            'bbox': [x1, y1, annw, annh],
                            'category_id': catnumid,
                            # 'info3d': info3d,
                            # 'W': np.log((annw * annos['location'][annidex, 2] / fx) / means[0]),
                            # 'H': np.log((annh * annos['location'][annidex, 2] / fy) / means[1]),
                            # 'D': np.log(annos['location'][annidex, 2] / means[2]),
                            # 'depth': annos['location'][annidex, :][2],
                            # 'location': annos['location'][annidex, :].tolist(),
                            # 'dimensions': annos['dimensions'][annidex, :].tolist(),
                            "truncated": float(annos['truncated'][annidex]),
                            "occluded": float(annos['occluded'][annidex]),
                            "rotation": annos['rotation_y'][annidex],
                        }
                        annotate.append(anntmp)

        annall = {
            "images": imgs,
            "type": "instance",
            "categories": cates,
            "annotations": annotate
        }

        with open(os.path.join(datatype + '.json'), 'w') as f:
            json.dump(annall, f)


if __name__ == '__main__':
    main()
