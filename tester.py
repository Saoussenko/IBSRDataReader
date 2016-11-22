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
clf.fit(x[:1120], y[:1120])

print 'fitting finshed'

predictions = clf.predict(x[1120:])

y = y[1120:]
for i in range(4):
    true_positive = np.sum(np.multiply(predictions == i, y == i))
    false_positive = np.sum(np.multiply(predictions == i, y != i))
    false_negative = np.sum(np.multiply(predictions != i, y == i))
    print 'class %d' % i
    print true_positive
    print false_positive
    print false_negative
    print 2 * true_positive / (2.0 * true_positive + false_positive + false_negative)
