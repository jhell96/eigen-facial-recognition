""" 
Josh Hellerstein
05/2018
"""


import cv2
import numpy as np
import os

from face_recognition.preprocess import *
from face_recognition.database2 import *

import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


class Eigenfaces():

    database = None
    all_faces = []
    mean_img = None
    weights = None
    evecs = None
    evals = None
    rf = None

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

        print("found covariance")

        evals, evecs = np.linalg.eig(C)
        inds = evals.argsort()[::-1]
        evals = evals[inds]
        evecs = evecs[:, inds]

        print("found eigs")

        i = 0
        cumulative_energy = 0
        total = sum(evals)
        while cumulative_energy < energy:
            cumulative_energy += (evals[i] / total)
            i += 1

        print("chosen top eigs")
        
        evals = evals[:i]
        evecs = evecs[:, :i]

        evecs = np.dot(T, evecs)
        norm = np.linalg.norm(evecs, axis=0)

        evecs = evecs / norm

        self.evecs = evecs
        self.evals = evals

        weights = np.dot(evecs.T, T)
        self.weights = weights
        print("found weights")

        self.rf = self.train_rf()
        self.inspect()

    def predict(self, img, metric='l2_norm', top=5):
        im = np.array(img, dtype='float64').flatten()
        im = im[:, np.newaxis]
        im -= self.mean_img
        weights = np.dot(self.evecs.T, im)

        if metric == 'l1_norm':
            distances = np.linalg.norm(self.weights - weights, axis=0, ord=1)
            closest = np.argsort(distances)
            return [self.all_faces[closest[i]][1] for i in np.arange(top)]

        elif metric == 'l2_norm':
            distances = np.linalg.norm(self.weights - weights, axis=0, ord=2)
            closest = np.argsort(distances)
            return [self.all_faces[closest[i]][1] for i in np.arange(top)]

        elif metric == 'rf':
            distances = self.rf[0].predict_proba([weights.flatten()])[0]
            closest = list(reversed(np.argsort(distances)))
            return [self.database.uid_to_name[self.rf[1][closest[i]]] for i in np.arange(top)]

    def evaluate(self, num_trials=5, metric='l2_norm', top=5):
        avg_precision =0

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


    def train_rf(self):
        x_train = self.weights.T

        ordered_uids = [x[2] for x in self.all_faces]
        
        uid_to_class = {}
        class_to_uid = {}

        for i, uid in enumerate(set(ordered_uids)):
            uid_to_class[uid] = i
            class_to_uid[i] = uid

        y_train = np.array([uid_to_class[x] for x in ordered_uids])

        # regr = RandomForestClassifier(max_depth=None, random_state=0)

        regr = SVC(probability=True, kernel='linear', gamma=0.0001, C=1000.0)
        
        regr = regr.fit(x_train, y_train)

        return regr, class_to_uid

    def inspect(self):

        # output top 10 eigenfaces

        # for i in range(10):
        #     face = map_to_image_range(self.evecs[:,i]).reshape(100,100)
        #     cv2.imwrite('../eigen_face'+str(i)+'.jpg', face)


        # cumulative variation captured by eigenvalues

        # total = sum(self.evals)
        # cumulative = 0
        # vals = [0.0]
        # for ev in self.evals:
        #     cumulative += ev/total
        #     vals.append(cumulative)

        # plt.plot(vals)
        # plt.title('Cumulative Energy Captured')
        # plt.xlabel('Number of Eigenvalues')
        # plt.ylabel('Percent Variation Captured')
        # plt.show()

        pass

    def show(self, img):
        cv2.imshow('img', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
