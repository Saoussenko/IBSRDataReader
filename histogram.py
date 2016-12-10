import numpy as np
import matplotlib.pyplot as plt


def equalize(f):
    h = np.histogram(f, bins=16384)[0]
    H = np.cumsum(h) / float(np.sum(h))
    import pdb
    pdb.set_trace()
    e = np.floor(H[(f.flatten() * 16383.).astype('int')])
    return e.reshape(f.shape)


def plt_histogram(image):
    import pdb
    pdb.set_trace()
    hist, bins = np.histogram(image, bins=256)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width)
    plt.show()

image_address = '/home/siavash/programming/thesis/final_data/images/1_24.npy'
image = np.load(image_address)

for i in xrange(image.shape[1]):
    image[:, i, :] = image[:, i, :] / np.max(image[:, i, :])

import pdb
pdb.set_trace()