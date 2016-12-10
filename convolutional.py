import random

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

images = np.load('images.npy')
labels = np.load('labels.npy')
labels = np.eye(4)[[labels.astype(int)]].reshape(labels.shape[0], 4)
shuffled_array = range(len(images))
random.shuffle(shuffled_array)
images = images[shuffled_array]
labels = labels[shuffled_array]
separator = int(len(images) * 0.7)
test_images = images[separator:]
test_labels = labels[separator:]
images = images[:separator]
labels = labels[:separator]

print 'shape of inputs: %s' % str(images.shape)
print 'shape of labels: %s' % str(labels.shape)

input_size = 11 * 1 * 11
output_size = 4

# mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
# input_size = 28 * 1 * 28
# output_size = 10

alpha = 1e-4
iterations = 400000


x_input = tf.placeholder("float", shape=[None, input_size])
y_ = tf.placeholder("float", shape=[None, output_size])


#MNIST
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x_input, [-1, 11, 11, 1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)


W_fc1 = weight_variable([3 * 3 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 3*3*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, output_size])
b_fc2 = bias_variable([output_size])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
train_step = tf.train.AdamOptimizer(alpha).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())
for i in range(iterations):
    start = i * 50 % separator
    end = start + 50
    batch = images[start:end], labels[start:end]
    # batch = mnist.train.next_batch(50)
    if i % 100 == 0:
        # print x_image.eval(feed_dict={x_input: batch[0]}).shape
        # print h_pool1.eval(feed_dict={x_input: batch[0]}).shape
        # print h_pool2.eval(feed_dict={x_input: batch[0]}).shape
        train_accuracy = accuracy.eval(feed_dict={
            x_input: batch[0], y_: batch[1], keep_prob: 1.0})
        print "step %d, training accuracy %g" % (i, train_accuracy)
        train_step.run(feed_dict={x_input: batch[0], y_: batch[1], keep_prob: 0.5})

print "test accuracy %g" % accuracy.eval(feed_dict={
    x_input: test_images, y_: test_labels, keep_prob: 1.0
})


# sess = tf.InteractiveSession()
#
# x = tf.placeholder("float", shape=[None, input_size])
# y_ = tf.placeholder("float", shape=[None, output_size])
#
# W = tf.Variable(tf.zeros([input_size, output_size]))
# b = tf.Variable(tf.zeros([output_size]))
#
# y = tf.nn.softmax(tf.matmul(x, W) + b)
# cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
#
# train_step = tf.train.AdamOptimizer().minimize(cross_entropy)



# print images.shape
# sess.run(tf.initialize_all_variables())
# for i in range(1000):
#     start = i * 50 % 1120
#     end = start + 50
#     batch = images[start:end], labels[start:end]
#     # batch = mnist.train.next_batch(50)
#     train_step.run(feed_dict={x: batch[0], y_: batch[1]})

# correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
# print accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels})
# print accuracy.eval(feed_dict={x: test_images, y_: test_labels})

predictions = tf.argmax(y_conv, 1)

predictions_np = predictions.eval(feed_dict={x_input: test_images, keep_prob: 1.0})

test_labels = np.argmax(test_labels, 1)
for i in range(4):
    true_positive = np.sum(np.multiply(predictions_np == i, test_labels == i))
    false_positive = np.sum(np.multiply(predictions_np == i, test_labels != i))
    false_negative = np.sum(np.multiply(predictions_np != i, test_labels == i))
    print 'class %d' % i
    print true_positive
    print false_positive
    print false_negative
    print 2 * true_positive / (2.0 * true_positive + false_positive + false_negative)


# image_address = '/home/siavash/programming/thesis/final_data/images/1_24.npy'
# image = np.load(image_address)
# image = image / np.max(image)
#
# lbl = np.zeros([256, image.shape[1], 256])
# points = [(x, y, z) for x in range(25, 231) for y in range(2, image.shape[1] - 3) for z in range(25, 231)]
#
# from generate_datasets import extract_point
#
# import pdb
# pdb.set_trace()
# counter = 0
# print len(points)
# for point in points:
#     counter += 1
#     img = extract_point(image, point)
#     img = img.reshape([1, reduce(lambda x, y: x * y, img.shape)])
#     guess = predictions.eval(feed_dict={x_input: img, keep_prob: 1.0})[0]
#     lbl[point] = predictions.eval(feed_dict={x_input: img, keep_prob: 1.0})[0]
#     if counter % 100 == 0:
#         print guess
#         print counter
#         print point
#
# np.save('lbl.npy', lbl)
