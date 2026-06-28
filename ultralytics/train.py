
import torch
from ultralytics import YOLO
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import argparse
import os

# H:\code\ultralytics\ultralytics\cfg\models\v8\yolov8.yaml

if __name__ == '__main__':
    # model = YOLO('./runs/detect/train135/weights/best.pt')
    model = YOLO('./mycfg/yolo26-lks-smoe-123.yaml')
    model.train(data='H:\\data_f\Visdrone_DET\\VisDrone2019-DET\\VisDrone2019-DET-yolo\\data.yaml', epochs=300, resume=True)
    result = model.val()
