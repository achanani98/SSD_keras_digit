import os
import cv2
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd

def read_sample(image_path, label_path):
    """ Read image and label file in xml format.

    Args:
        - image_path: path to image file
        - label_path: path to label xml file

    Returns:
        - image: a numpy array with a data type of float
        - bboxes: a numpy array with a data type of float
        - classes: a list of strings

    Raises:
        - Image file does not exist
        - Label file does not exist
    """
    image_path = image_path.strip("\n")
    label_path = label_path.strip("\n")
    # print(image_path,label_path)
    # print(os.getcwd())
    assert os.path.exists(image_path), "Image file does not exist."
    assert os.path.exists(label_path), "Label file does not exist."

    image = cv2.imread(image_path)  # read image in bgr format
    bboxes, classes = [], []
    
    with open(label_path,'r+') as f:
        labels = (f.readline()).split(" ")
        name = labels[0]
        xmin = labels[1]
        ymin = labels[2]
        xmax = labels[3]
        ymax = labels[4]
        bboxes.append([xmin,ymin,xmax,ymax])
        classes.append(name)
    return np.array(image, dtype=np.float), np.array(bboxes, dtype=np.float), classes

