""" 
Josh Hellerstein
05/2018
"""


import cv2
import numpy as np
import os

curr_path = os.path.dirname(__file__)
model_path = os.path.join(curr_path, "../data/models/lbpcascade_frontalface.xml")

def detect_face(img, pad=0):
    face_cascade = cv2.CascadeClassifier(model_path)
    faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=10)

    if (len(faces) == 0):
        return img, []

    (x, y, w, h) = faces[0]

    y_pad = int(w*pad)
    x_pad = int(h*pad)
    return img[y-y_pad:y+w+y_pad, x-x_pad:x+h+x_pad], faces[0]

def preprocess(img, img_size_out=(100,100)):
    img_face = detect_face(img)[0]

    img = cv2.resize(img_face, dsize=img_size_out, interpolation=cv2.INTER_AREA)

    img = cv2.equalizeHist(img)
    return img