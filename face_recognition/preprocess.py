""" 
Josh Hellerstein
05/2018
"""


import cv2
import os
import numpy as np

curr_path = os.path.dirname(__file__)
model_path = os.path.join(curr_path, "../data/models/lbpcascade_frontalface.xml")

def detect_face(img):
    face_cascade = cv2.CascadeClassifier(model_path)
    faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)

    if (len(faces) == 0):
        return None, None

    (x, y, w, h) = faces[0]
    return img[y:y+w, x:x+h], faces[0]

def read_img(file):
    im = cv2.imread(file, 0)
    im = cv2.equalizeHist(im)
    return im

def show_img(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()