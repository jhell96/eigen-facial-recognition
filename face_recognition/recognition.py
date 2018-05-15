""" 
Josh Hellerstein
05/2018
"""


import cv2
import numpy as np
import os

from face_recognition.preprocess import *
from face_recognition.database import *

class Eigenfaces():

    database = None
    mean_img = None
    weights = None
    evecs = None
    evals = None

    def __init__(self, database, energy=0.85):
        self.database = database
        n = len(database.all_faces)
        T = np.empty(shape=(300*300, n), dtype='float64')

        for i, face in enumerate(database.all_faces):
            im = np.array(face[0], dtype='float64').flatten()
            T[:, i] = im[:]

        mean_img = np.sum(T, axis=1) / n
        self.mean_img = mean_img

        for i in range(n):
            T[:, i] -= mean_img

        C = np.dot(T.T, T) / n

        evals, evecs = np.linalg.eig(C)
        inds = evals.argsort()[::-1]
        evals = evals[inds]
        evecs = evecs[inds]

        i = 0
        cumulative_energy = 0
        total = sum(evals)
        while cumulative_energy < energy:
            cumulative_energy += (evals[i] / total)
            i += 1

        evals = evals[:i]
        evecs = evecs[:i]

        evecs = evecs.T
        evecs = np.dot(T, evecs)
        evecs = evecs / np.linalg.norm(evecs, axis=0)

        self.evecs = evecs
        self.evals = evals

        weights = np.dot(evecs.T, T)
        self.weights = weights

    def predict(self, img):
        im = np.array(img, dtype='float64').flatten()
        im -= self.mean_img
        im = im[:, np.newaxis]
        weights = np.dot(self.evecs.T, im)

        distances = np.linalg.norm(self.weights - weights, axis=0)
        closest = np.argmin(distances)

        return self.database.all_faces[closest][1].name