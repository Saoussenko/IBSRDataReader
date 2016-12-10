import random

import numpy as np
from sklearn.neural_network import MLPClassifier


x = np.load('images.npy')
y = np.load('labels.npy')

shuffled_indices = range(x.shape[0])
random.shuffle(shuffled_indices)

x = x[shuffled_indices]
y = y[shuffled_indices]


clf = MLPClassifier(solver='adam', alpha=1e-2, hidden_layer_sizes=(1000), random_state=1, verbose=True)
clf.fit(x[:4480], y[:4480])

print 'fitting finshed'

predictions = clf.predict(x[4480:])

y = y[4480:]
for i in range(4):
    true_positive = np.sum(np.multiply(predictions == i, y == i))
    false_positive = np.sum(np.multiply(predictions == i, y != i))
    false_negative = np.sum(np.multiply(predictions != i, y == i))
    print 'class %d' % i
    print true_positive
    print false_positive
    print false_negative
    print 2 * true_positive / (2.0 * true_positive + false_positive + false_negative)

exit()
image_address = '/home/siavash/programming/thesis/final_data/images/1_24.npy'
image = np.load(image_address)
image = image / np.max(image)

lbl = np.zeros([256, image.shape[1], 256])
points = [(x, y, z) for x in range(25, 231) for y in range(2, image.shape[1] - 3) for z in range(25, 231)]

from generate_datasets import extract_point

import pdb
pdb.set_trace()
counter = 0
print len(points)
for point in points:
    counter += 1
    if counter % 100 == 0:
        print counter
        print point
    img = extract_point(image, point)
    img = img.reshape([1, reduce(lambda x, y: x * y, img.shape)])
    lbl[point[0], point[1], point[2]] = clf.predict(img)

np.save('lbl.npy', lbl)
