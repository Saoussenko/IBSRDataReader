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
    0: 80,
    1: 80,
    2: 80,
    3: 80,
}

# The dimension are something like as: 256, 50, 256
cuboid_dimensions = 11, 1, 11
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


def generate_datasets(prefix):
    images = []
    labels = []
    for file_address in os.listdir(data_folder + prefix + 'images/'):
        image = np.load(data_folder + prefix + 'images/' + file_address)
        # image = image / np.max(image)
        for i in xrange(image.shape[1]):
            if np.max(image[:, i, :]) != 0:
                image[:, i, :] = image[:, i, :] / np.max(image[:, i, :])
        label = np.load(data_folder + prefix + 'labels/' + file_address[:-4] + '_lbl.npy')
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

    shuffled_indices = range(len(images))
    np.random.shuffle(shuffled_indices)
    images = images[shuffled_indices]
    labels = labels[shuffled_indices]

    return images, labels, shuffled_indices

required_number_of_points_for_each_class = {
    0: 80,
    1: 80,
    2: 80,
    3: 80,
}

tr_x, tr_y, tr_shuffled_indices = generate_datasets('tr_')
np.save('tr_images.npy', tr_x)
np.save('tr_labels', tr_y)
np.save('tr_shuffled_indices', tr_shuffled_indices)


required_number_of_points_for_each_class = {
    0: 80,
    1: 80,
    2: 80,
    3: 80,
}
te_x, te_y, te_shuffled_indices = generate_datasets('te_')
np.save('te_images.npy', te_x)
np.save('te_labels', te_y)
np.save('te_shuffled_indices', te_shuffled_indices)