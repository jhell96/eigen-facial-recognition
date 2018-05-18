""" 
Josh Hellerstein
05/2018
"""

import os
import enroll

if __name__ == '__main__':

    # path to LFW (labeled faces in the wild) dataset
    path_to_lfw_db = "../josh_db"

    names = os.listdir(path_to_lfw_db)

    # only take 100 people
    limit = 100
    for name in names:
        path = os.path.join(path_to_lfw_db, name)

        if os.path.isdir(path):
            image_names = os.listdir(path)

            # take 3 images from each person
            num_to_take = 3
            if len(image_names) >= num_to_take:
                image_paths = [os.path.join(path, image_name) for image_name in image_names[:num_to_take]]

                # call our enroll function to add the person to the database
                enroll.enroll_faces(name, image_paths)

                if limit > 0:
                    limit -= 1
                else:
                    break



