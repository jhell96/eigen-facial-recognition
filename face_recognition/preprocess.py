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
    faces = face_cascade.detectMultiScale(img, scaleFactor=1.05, minNeighbors=10)

    if (len(faces) == 0):
        return img, []

    (x, y, w, h) = faces[0]
    return img[y:y+w, x:x+h], faces[0]

def preprocess(img, img_size_out=(100,100)):
    img_face = detect_face(img)[0]

    img = cv2.resize(img_face, dsize=img_size_out, interpolation=cv2.INTER_CUBIC)

    img = cv2.equalizeHist(img)
    return img