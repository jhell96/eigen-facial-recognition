""" 
Josh Hellerstein
05/2018
"""


import cv2
import numpy as np
import os

from face_recognition.preprocess import *
from face_recognition.database2 import *


class Eigenfaces():

    database = None
    all_faces = []
    mean_img = None
    weights = None
    evecs = None
    evals = None

    def __init__(self, database, energy=0.85):

        self.database = database
        self.all_faces = database.get_all_faces()
        n = len(self.all_faces)
        shape = self.all_faces[0][0].shape
        dim = shape[0] * shape[1]

        T = np.empty(shape=(dim, n), dtype='float64')

        for i, face in enumerate(self.all_faces):
            im = np.array(face[0], dtype='float64').flatten()
            T[:, i] = im[:]

        print("flattened")

        mean_img = T.mean(axis=1, keepdims=True)
        self.mean_img = mean_img

        T -= mean_img
        
        print("subtracted mean")

        C = np.dot(T.T, T) / n

        print("multiplied")

        evals, evecs = np.linalg.eig(C)
        inds = evals.argsort()[::-1]
        evals = evals[inds]
        evecs = evecs[inds]

        print("found eigs")

        i = 0
        cumulative_energy = 0
        total = sum(evals)
        while cumulative_energy < energy:
            cumulative_energy += (evals[i] / total)
            i += 1

        print("chosen top eigs")
        
        evals = evals[:i]
        evecs = evecs[:i]

        evecs = evecs.T
        evecs = np.dot(T, evecs)
        evecs = evecs / np.linalg.norm(evecs, axis=0)

        self.evecs = evecs
        self.evals = evals

        weights = np.dot(evecs.T, T)
        self.weights = weights
        print("found weights")

    def predict(self, img):
        im = np.array(img, dtype='float64').flatten()
        im = im[:, np.newaxis]
        im -= self.mean_img
        weights = np.dot(self.evecs.T, im)

        distances = np.linalg.norm(self.weights - weights, axis=0)
        closest = reversed(np.argsort(distances))

        return self.all_faces[closest][1]