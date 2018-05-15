""" 
Josh Hellerstein
05/2018
"""


import random
import time
import numpy as np

class Person():
    name = None
    faces = []
    uid = -1
    created = -1

    def __init__(self, name, img):
        self.name = name
        self.uid = str(random.randint(0, 99999999999))
        self.created = time.time()

        self.add_face(img)

    def add_face(self, img):
        self.faces.append(img)

    def get_faces(self):
        return self.faces

    def __getstate__(self):
        """Custom function for pickling images of faces,
            along with the Person object
        
        Returns:
            TYPE: object's interal dict
        """
        self.faces = [(f.tostring(), f.shape) for f in self.faces]
        state = self.__dict__.copy()
        return state

    def __setstate__(self, newstate):
        """Custom function for unpickling images of faces
        
        Args:
            newstate (dict): Object's previous state, as its internal dict
        """
        self.__dict__.update(newstate)
        faces = newstate['faces']
        self.faces = [np.fromstring(f[0], dtype=np.uint8).reshape(f[1]) for f in faces]
