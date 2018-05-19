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
    cv2.imwrite('face1.jpg', img)

    img_face = detect_face(img)[0]

    cv2.imwrite('face2.jpg', img_face)

    img = cv2.resize(img_face, dsize=img_size_out, interpolation=cv2.INTER_AREA)

    cv2.imwrite('face3.jpg', img)

    img = cv2.equalizeHist(img)

    cv2.imwrite('face4.jpg', img)
    return img

def map_to_image_range(old_img):
    old_range = (np.max(old_img) - np.min(old_img))
    new_range = (255.0-0.0)
    new_image = (((old_img - np.min(old_img)) * new_range) / old_range)

    return np.array(new_image, dtype=np.uint8)
