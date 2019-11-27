import tensorflow as tf
import numpy as np
import config
import load
import matplotlib.pyplot as plt

learning_rate = 1e-4


def normalize(x):
    x = np.array(x)
    return (x - x.min()) / (np.ptp(x))

# load data
X_train, y_train, X_test, y_test = load.load_sample_dataset()

# flatten to one vector of 28x28 images
X_train = list(map(lambda x: x.flatten(), X_train))
X_test = list(map(lambda x: x.flatten(), X_test))


# one hot encode labels
characters = np.max(y_train) + 1
y_train = (lambda l: [item for sublist in l for item in sublist])(y_train)
y_test = (lambda l: [item for sublist in l for item in sublist])(y_test)
y_train = np.eye(characters)[y_train]
y_test = np.eye(characters)[y_test]


# normalize between 0-1 from 0-255
X_train = np.array(X_train) / np.std(X_train)
X_test = np.array(X_test) / np.std(X_test)
X_train = normalize(X_train)
X_test = normalize(X_test)


total_mean = np.mean(np.array(list(X_train) + list(X_test)))

# remove mean to get zero mean for each image
X_train = X_train - total_mean
X_test = X_test - total_mean

X = tf.placeholder(tf.float32, [None, X_train.shape[1]], name='Inputs',)
y = tf.placeholder(tf.float32, [None, characters], name='Labels')

with tf.variable_scope('svm'):
    W = tf.get_variable('W', [X_train.shape[1], characters], initializer=tf.contrib.layers.xavier_initializer())

    b = tf.get_variable('b', [characters], initializer=tf.initializers.constant(0.1))

    y_logits = tf.matmul(X, W) + b

    cost = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_logits)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)


def train():
    with tf.Session() as sess:
        epochs = 0
        total_epochs = 500

        cost_history = []
        accuracy_history = []

        gvi = tf.global_variables_initializer()
        sess.run(gvi)
        while True:
            feed_dict = {X: X_train, y: y_train}
            sess.run(optimizer, feed_dict=feed_dict)
            current_cost = sess.run(cost, feed_dict=feed_dict)
            epochs += 1
            average_cost = np.average(current_cost)
            print('average loss: {0}'.format(average_cost))
            print('Epoch {0}'.format(epochs))

            correct_prediction = tf.equal(tf.argmax(y_logits, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            cost_history.append(average_cost)
            accuracy_history.append(accuracy.eval({X: X_train, y: y_train}) * 100)

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
        saver.save(sess=sess, save_path="models/21_superrene.cpkt")


def image_reshaper(model_path):
    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, model_path)
        class_features = W.eval()
        index = 1

        for image_class in np.transpose(class_features):
            plt.subplot(2, 5, index)
            image = normalize(image_class) * 255
            image = image.reshape((28, 28))
            plt.title('Image {0}'.format(index))
            plt.axis('off')
            plt.imshow(image, cmap='gray', vmin=0, vmax=255)
            index += 1
        plt.show()


#image_reshaper("models/21_superrene.cpkt")
train()
