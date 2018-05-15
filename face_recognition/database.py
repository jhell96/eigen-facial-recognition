""" 
Josh Hellerstein
05/2018
"""

import os
import pickle

from face_recognition.person import *


class Database():
    curr_path = os.path.dirname(__file__)
    people_path = os.path.join(curr_path, "../data/people")
    people = []
    people_ids = {}
    all_faces = []

    def __init__(self):
        self.load_people()

    def load_people(self):
        people_files = os.listdir(self.people_path)
        
        for p_file in people_files:
            full_path = os.path.join(self.people_path, p_file)

            if os.path.getsize(full_path) > 0 and os.path.splitext(full_path)[1] == '.pkl':
                with open(full_path, 'rb') as f:
                    p = pickle.load(f)
                    self.people.append(p)
                    self.people_ids[p.uid] = p
                    self.all_faces.extend([(f, p) for f in p.get_faces()])

    def create_person(self, name, img):
        name = str(name)

        p = Person(name, img)

        if p.uid not in self.people_ids.keys():
            self.people.append(p)
            self.people_ids[p.uid] = p
            self.all_faces.extend([(f, p) for f in p.get_faces()])
            with open(os.path.join(self.people_path, p.uid + '.pkl'), 'wb') as f:
                pickle.dump(p, f)

        else:
            raise ValueError('Person already exists in DB!')

    def update_person(self, uid, img):
        uid = str(uid)

        if uid in self.people_ids.keys():
            p = self.people_ids[uid]
            p.add_face(img)
            self.all_faces.append((img, p))
            with open(os.path.join(self.people_path, uid + '.pkl'), 'wb') as f:
                pickle.dump(p, f)

        else:
            raise ValueError('Person not in DB but trying to update person!')

    def get_person(self, uid):
        uid = str(uid)

        if uid in self.people_ids.keys():
            return self.people_ids[uid]
        
        else:
            return None

    def get_people(self):
        return self.people
