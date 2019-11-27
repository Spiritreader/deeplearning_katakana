#import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
import numpy as np
import config
import load
import sys
import matplotlib.pyplot as plt


learning_rate = 1e-4


def normalize(x):
    x = np.array(x)
    return (x - x.min()) / (np.ptp(x))


# load data
X_train, y_train, X_test, y_test = load.load_sample_dataset()

X_train = np.expand_dims(X_train, axis=3)
X_test = np.expand_dims(X_test, axis=3)

# flatten to one vector of 28x28 images
# X_train = list(map(lambda x: x.flatten(), X_train))
# X_test = list(map(lambda x: x.flatten(), X_test))


# one hot encode labels
characters = np.max(y_train) + 1
y_train = (lambda l: [item for sublist in l for item in sublist])(y_train)
y_test = (lambda l: [item for sublist in l for item in sublist])(y_test)
y_train = np.eye(characters)[y_train]
y_test = np.eye(characters)[y_test]

total_mean = np.mean(np.array(list(X_train) + list(X_test)))

# remove mean to get zero mean for each image
X_train = X_train - total_mean
X_test = X_test - total_mean


# normalize between 0-1 from 0-255
X_train = np.array(X_train) / np.std(X_train)
X_test = np.array(X_test) / np.std(X_test)
X_train = normalize(X_train)
X_test = normalize(X_test)


X = tf.placeholder(tf.float32, [None, X_train.shape[1], X_train.shape[2], 1], name='Inputs',)
y = tf.placeholder(tf.float32, [None, characters], name='Labels')


def max_pool_2x2(input_layer):
    return tf.nn.max_pool(input_layer, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def conv2d_layer(name, x_dim, y_dim, input_dim, filters, input_layer):
    """
    :param name: Name of the layer
    :param x_dim: x dimension of the 2D image
    :param y_dim: y dimension of the 2D image
    :param input_dim: input dimension (greyscale, rgb, etc)
    :param filters: amount of filters for the convolution
    :param input_layer: the input layer to connect to
    """
    with tf.variable_scope(('conv2d_' + name)):
        W = tf.get_variable(name, [x_dim, y_dim, input_dim, filters], initializer=tf.contrib.layers.xavier_initializer())
        c = tf.nn.conv2d(input_layer, W, strides=[1, 1, 1, 1], padding='SAME')
        b = tf.get_variable('b_{0}'.format(name), [filters])
        hc1 = tf.nn.leaky_relu(c + b)
        return hc1


def dense_layer(name, dense_features, input_layer, classifiers):
    with tf.variable_scope('dense_' + name):
        classifier_in = tf.reshape(input_layer, [-1, dense_features])
        W = tf.get_variable(name, [dense_features, classifiers], initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable('b', classifiers)
        node = tf.nn.leaky_relu(tf.matmul(classifier_in, W) + b)
        return node


conv_0_filters = 32
conv_1_filters = 16
conv_2_filters = 16
dense_0_nodes = 16
conv_0 = conv2d_layer("0", 5, 5, 1, conv_0_filters, X)
max_pool_0 = max_pool_2x2(conv_0)
conv_1 = conv2d_layer("1", 3, 3, conv_0_filters, conv_1_filters, max_pool_0)
max_pool_1 = max_pool_2x2(conv_1)
conv_2 = conv2d_layer("2", 2, 2,conv_1_filters, conv_2_filters, max_pool_1)
dense_0 = dense_layer("0", 7*7*conv_2_filters, conv_2, dense_0_nodes)
y_logits = dense_layer("1", dense_0_nodes, dense_0, characters)

cost = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_logits)

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# IT JUST WORKS!!!!
# optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)


def train():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        epochs = 0
        total_epochs = 80

        cost_history = []
        accuracy_history = []

        gvi = tf.global_variables_initializer()
        sess.run(gvi)
        train_batch_split = 100
        X_train_chunked = np.array_split(X_train, train_batch_split)
        X_train_chunked = [x for x in X_train_chunked if x.size > 0]
        y_train_chunked = np.array_split(y_train, train_batch_split)
        y_train_chunked = [x for x in y_train_chunked if x.size > 0]
        while True:
            iteration = 0
            for i in range(0, len(X_train_chunked)):
                feed_dict = {X:X_train_chunked[i], y: y_train_chunked[i]}
                sess.run(optimizer, feed_dict=feed_dict)
                iteration += 1

            current_cost = sess.run(cost, feed_dict=feed_dict)
            average_cost = np.average(current_cost)
            correct_prediction = tf.equal(tf.argmax(y_logits, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            acc_eval = accuracy.eval({X: X_train, y: y_train}) * 100

            cost_history.append(average_cost)
            accuracy_history.append(acc_eval)
            print('average loss: {0}'.format(average_cost))
            print('Epoch {0}'.format(epochs))
            print('training accuracy: {0}'.format(acc_eval))
            epochs += 1
            if epochs == total_epochs:
                break

        fig, ax1 = plt.subplots()

        color = 'tab:red'
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Cost', color=color)
        ax1.plot(list(range(epochs)), cost_history, color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        color = 'tab:blue'
        ax2.set_ylabel('Training Accuracy', color=color)  # we already handled the x-label with ax1
        ax2.plot(list(range(epochs)), accuracy_history, color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.show()

        correct_prediction = tf.equal(tf.argmax(y_logits, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        evaluated_accuracy = accuracy.eval({X: X_test, y: y_test}) * 100
        print("Test accuracy: {0}%".format(evaluated_accuracy))
        saver = tf.train.Saver()
        saver.save(sess=sess, save_path="models/22_cnn.cpkt")

# image_reshaper("models/21_superrene.cpkt")
train()
