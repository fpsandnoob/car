# http://blog.csdn.net/sunbaigui/article/details/39938097
from uitil import tic, toc
import tensorflow as tf
import data_input
import shutil

x = tf.placeholder("float", shape=[None, 227, 227, 3])
y = tf.placeholder("float", shape=[None, 196])
keep_prob = tf.placeholder("float")
d = data_input.Data()
d.get_img()

with tf.name_scope('conv1') as scope:
    weights1 = tf.Variable(tf.truncated_normal([11, 11, 3, 96]), name='weights1')
    biases1 = tf.Variable(tf.truncated_normal([96]), name='biases2')
    conv1 = tf.nn.conv2d(x,
                         filter=weights1, strides=[1, 4, 4, 1],
                         padding='SAME', name='conv1')
    conv1_relu = tf.nn.relu(conv1 + biases1, name='relu1')
    pool1 = tf.nn.max_pool(conv1_relu,
                           ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='maxpool1')
    norm1 = tf.nn.local_response_normalization(pool1, depth_radius=5, name='LRN1')

with tf.name_scope('conv2') as scope_1:
    weights2 = tf.Variable(tf.truncated_normal([3, 3, 96, 384]), name='weights2')
    biases2 = tf.Variable(tf.truncated_normal([384]), name='biases2')
    conv2 = tf.nn.conv2d(norm1,
                         filter=weights2, strides=[1, 1, 1, 1],
                         padding='VALID', name='conv2')
    conv2_relu = tf.nn.relu(conv2 + biases2, name='relu2')
    pool2 = tf.nn.max_pool(conv2_relu,
                           ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='maxpool2')
    norm2 = tf.nn.local_response_normalization(pool2, depth_radius=5, name='LRN2')
