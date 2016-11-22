import os
import random

import numpy as np


label_mapping = {
    0: 0,
    128: 1,
    192: 2,
    254: 3,
}
required_number_of_points_for_each_class = {
    0: 20,
    1: 20,
    2: 20,
    3: 20,
}

# The dimension are something like as: 256, 50, 256
cuboid_dimensions = 50, 5, 50
x_step = cuboid_dimensions[0] / 2
y_step = cuboid_dimensions[1] / 2
z_step = cuboid_dimensions[2] / 2

data_folder = '/home/siavash/programming/thesis/final_data/'


def extract_point(image, point):
    return image[
        point[0] - x_step:point[0] + x_step + 1,
        point[1] - y_step:point[1] + y_step + 1,
        point[2] - z_step:point[2] + z_step + 1,
    ]


def generate_datasets():
    images = []
    labels = []
    for file_address in os.listdir(data_folder + 'images/'):
        image = np.load(data_folder + 'images/' + file_address)
        image = image / np.max(image)
        label = np.load(data_folder + 'labels/' + file_address[:-4] + '_lbl.npy')
        for l, ml in label_mapping.iteritems():
            label[np.where(label == l)] = ml

        for l in label_mapping.values():
            points = np.argwhere(label == l)

            invalid_rows = np.where(points[:, 0] < x_step)[0]
            invalid_rows = np.append(invalid_rows, np.where(points[:, 0] >= image.shape[0] - x_step)[0])

            invalid_rows = np.append(invalid_rows, np.where(points[:, 1] < y_step)[0])
            invalid_rows = np.append(invalid_rows, np.where(points[:, 1] >= image.shape[1] - y_step)[0])

            invalid_rows = np.append(invalid_rows, np.where(points[:, 2] < z_step)[0])
            invalid_rows = np.append(invalid_rows, np.where(points[:, 2] >= image.shape[2] - z_step)[0])

            points = np.delete(points, invalid_rows, axis=0)

            number_of_points = len(points)
            random_indices = random.sample(xrange(number_of_points), required_number_of_points_for_each_class[l])
            for i in random_indices:
                random_point = points[i, :]
                img = extract_point(image, random_point)
                img = img.reshape([1, reduce(lambda x, y: x * y, img.shape)])
                images.append(img)
                labels.append(l)

    images = np.concatenate(images)
    labels = np.array(labels)
    return images, labels


tr_x, tr_y = generate_datasets()
np.save('images.npy', tr_x)
np.save('labels', tr_y)
