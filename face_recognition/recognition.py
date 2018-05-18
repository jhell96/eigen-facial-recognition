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

    def __init__(self, database, energy=0.90):
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
        norm = np.linalg.norm(evecs, axis=0)

        # show eigenvectors
        # cv2.imshow('im', evecs[:,0].reshape(100,100))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        evecs = evecs / norm

        self.evecs = evecs
        self.evals = evals

        weights = np.dot(evecs.T, T)
        self.weights = weights
        print("found weights")

    def predict(self, img, metric='l2_norm', top=5):
        im = np.array(img, dtype='float64').flatten()
        im = im[:, np.newaxis]
        im -= self.mean_img
        weights = np.dot(self.evecs.T, im)
        distances = self.distance_metric(weights, metric)
        closest = np.argsort(distances)

        return [self.all_faces[closest[i]][1] for i in np.arange(top)]

    def distance_metric(self, weights, metric):
        distances = []
        if metric == 'l1_norm':
            distances = np.linalg.norm(self.weights - weights, axis=0, ord=1)
        elif metric == 'l2_norm':
            distances = np.linalg.norm(self.weights - weights, axis=0, ord=2)

        return distances

    def evaluate(self, num_trials=5, metric='l2_norm', top=1):
        avg_precision = 0

        for i in range(num_trials):
            precision = 0

            faces = self.database.get_test_faces()
            n = len(faces)

            for face in faces:
                img, name = face[0], face[1]
                img = preprocess(img)
                res = self.predict(img, metric, top)
                if name.lower() in [x.lower() for x in res]:
                    precision += 1

            precision /= n
            avg_precision += precision

        return avg_precision/num_trials
