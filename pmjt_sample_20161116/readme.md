# Pre-modern Japanese Text Character Shape Dataset 
# Sample code for deep learning

## Summary
This is a sample code for character recognition on [Pre-modern Japanese Text Character Shape Dataset](http://codh.rois.ac.jp/char-shape/).

The code learns to recognize 10 most frequent characters in the dataset using a deep learning algorithm (convolutional neural network). 

The code is written in Python, and built upon the deep learning library Keras. About Keras, see [keras.io](https://keras.io/).

## Usage

python run.py

* To change basic settings, see config.py.
* To change data loading, see load.py. 
* Sampled data set is train_test_file_list.h5

## Description

1. Collect image data of 10 most frequent characters in the dataset. The total number of images is 23,423.
2. Divide the dataset into training (85% = 19,909 samples) and testing (15% = 3,514 samples).
3. Resize the image into 28*28 pixels and convert color to grey scale, and scaled the image by dividing 255.
4. Command line $ python run.py
5. Get 94.02% test accuracy after 12 epochs.

## Note

* The network architecture is based on the sample code from 
[https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py](https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py)
* The purpose of this sample code is to provide a baseline classifier.
* There is still a lot of margin for tuning. 

Last updated on November 16, 2016.
