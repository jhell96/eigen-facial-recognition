""" 
Josh Hellerstein
05/2018
"""

from face_recognition.preprocess import *
from face_recognition.recognition import *

if __name__ == "__main__":
    img = read_img("data/faces/ICA.jpg")

    f = detect_face(img)[0]
    show_img(f)
