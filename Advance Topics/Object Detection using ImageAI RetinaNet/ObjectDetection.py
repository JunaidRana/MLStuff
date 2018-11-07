# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 08:57:44 2018

@author: Junaid Rana
"""
"""
Find the Tutorial on given link to install dependencies and download RetinaNet

https://towardsdatascience.com/object-detection-with-10-lines-of-code-d6cb4d86f606

"""

from imageai.Detection import ObjectDetection
import os

execution_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()
#Take care of image extension
detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "image.jpeg"), output_image_path=os.path.join(execution_path , "imagenew.jpeg"))

for eachObject in detections:
    print(eachObject["name"] , " : " , eachObject["percentage_probability"] )