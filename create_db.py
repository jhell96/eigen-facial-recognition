""" 
Josh Hellerstein
05/2018
"""

import os
import enroll

if __name__ == '__main__':
    path_to_lfw_db = "../lfw"

    names = os.listdir(path_to_lfw_db)

    for name in names:
        path = os.path.join(path_to_lfw_db, name)
        image_names = os.listdir(path)

        image_paths = [os.path.join(path, image_name) for image_name in image_names]

        enroll.enroll_faces(name, image_paths)



