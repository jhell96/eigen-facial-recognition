""" 
Josh Hellerstein
05/2018
"""


import os
import argparse

from face_recognition.database import *
from face_recognition.preprocess import *


def read_img(file):
    im = cv2.imread(file, 0)
    return im

def enroll_face(img):
    d = Database()
    img = read_img(img)
    img = preprocess(img)
    uid = d.create_person(args.name, img)
    return uid

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enroll \
        a person or another picture of a person in the database")

    parser.add_argument("-i", "--img", help="The path to the image to add", required=True)
    parser.add_argument("-n", "--name", help="The name of the person", required=True)

    args = parser.parse_args()

    if os.path.exists(args.img):
        uid = enroll_face(args.img)    
        print(uid, "created")
    else:
        print("Please enter a valid path to an image, and name of the person to enroll")
    