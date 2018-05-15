""" 
Josh Hellerstein
05/2018
"""


import cv2
import numpy as np
import os

curr_path = os.path.dirname(__file__)
model_path = os.path.join(curr_path, "../data/models/lbpcascade_frontalface.xml")

def detect_face(img):
    face_cascade = cv2.CascadeClassifier(model_path)
    faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)

    if (len(faces) == 0):
        return None, None

    (x, y, w, h) = faces[0]
    return img[y:y+w, x:x+h], faces[0]

def preprocess(img, img_size_out=(300,300)):
    img = detect_face(img)[0]
    img = cv2.resize(img, dsize=img_size_out, interpolation=cv2.INTER_CUBIC)
    img = cv2.equalizeHist(img)
    return img