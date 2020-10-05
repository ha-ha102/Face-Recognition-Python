# -*- coding: utf-8 -*-
import numpy as np
import os
from os import listdir
from os.path import isfile, join
import pathlib
import cv2
import glob
from glob import iglob
from PIL import Image
import time
from tqdm import tqdm

mypath = "C:\\Users\\Hari\\Desktop\\FaceRecog\\TestData\\"
facedata = "C:\\Users\\Hari\\Desktop\\FaceRecog\\haarcascade_frontalface_default.xml"
cascade = cv2.CascadeClassifier(facedata)

file_list = [f for f in iglob('C:\\Users\\Hari\\Desktop\\FaceRecog\\TestData\\**\\*', recursive=True) if os.path.isfile(f)]
X = []
for f in tqdm(file_list):
    image = cv2.imread(f)
    minisize = (image.shape[1],image.shape[0])
    miniframe = cv2.resize(image, minisize)
    faces = cascade.detectMultiScale(miniframe)
    for b in faces:
        x, y, w, h = [ v for v in b ]
        crop_img = image[y:y+h, x:x+w]
    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    X.append(gray)
    
X_dat = []
hog = cv2.HOGDescriptor()   
for i in tqdm(X):
    h = hog.compute(i)
    X_dat.append(h)
    


    
    
    

