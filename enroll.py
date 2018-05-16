""" 
Josh Hellerstein
05/2018
"""


import os
import argparse

from face_recognition.database2 import *
from face_recognition.preprocess import *


def read_img(file):
    im = cv2.imread(file, 0)
    return im

def enroll_faces(name, imgs):
    d = Database()
    
    processed_imgs = []
    for img in imgs:
        img = read_img(img)
        img = preprocess(img)
        processed_imgs.append(img)

    uid = d.create_person(name, processed_imgs)
    return uid

def update_faces(uid, imgs):
    d = Database()

    processed_imgs = []
    for img in imgs:
        img = read_img(img)
        img = preprocess(img)
        processed_imgs.append(img)

    d.update_person(uid, processed_imgs)
    return len(d.get_faces(uid))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enroll \
        a person or another picture of a person in the database")

    parser.add_argument("-i", "--img", help="The path to the image to add", required=True)
    parser.add_argument("-n", "--name", help="The name of the person", required=False, default="")
    parser.add_argument("-u", "--update", help="The uid of the person to add an image to", required=False, default=None)

    args = parser.parse_args()

    if os.path.exists(args.img):
        if args.update and len(args.update) > 0:
            num_images = update_faces(args.update, [args.img])
            print(num_images, "images exist for", args.update)
        else:
            uid = enroll_faces(args.name, [args.img])    
            print(uid, "created")
    else:
        print("Please enter a valid path to an image, and name of the person to enroll")
    