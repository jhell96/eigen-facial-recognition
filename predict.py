""" 
Josh Hellerstein
05/2018
"""


import cv2

from face_recognition.preprocess import *
from face_recognition.recognition import *
from face_recognition.database import *


def read_img(file):
    im = cv2.imread(file, 0)
    return im

def show_img(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    d = Database()

    # d.get_people()

    img = read_img("../bp2.jpg")
    img = preprocess(img)
    # show_img(img)
    
    # d.update_person(52468199568, img)
    # d.create_person("brad pitt", img)

    # img = d.get_people()[0].faces[1]
    # show_img(img)

    # f = detect_face(img)[0]
    # show_img(f)

    e = Eigenfaces(d)
    res = e.predict(img)
    print(res)
