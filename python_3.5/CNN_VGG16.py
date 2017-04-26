import data_input
import tensorflow as tf

x = tf.placeholder("float", shape=[None, 224, 224, 3])
y = tf.placeholder("float", shape=[None, 1, 196])
keep_prob = tf.placeholder("float")
d = data_input.Data()
d.get_img()


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv(x, _in, _out):
    return tf.nn.relu(tf.nn.conv2d(x, weight_variable([3, 3, _in, _out]), strides=[1, 1, 1, 1], padding="SAME") +
                      bias_variable([_out]))


def maxpool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


W_fc1 = weight_variable([7 * 7 * 512, 4096])
b_fc1 = bias_variable([4096])
W_fc2 = weight_variable([4096, 4096])
b_fc2 = bias_variable([4096])
W_fc3 = weight_variable([4096, 1024])
b_fc3 = bias_variable([1024])
W_softmax = weight_variable([1024, 196])
b_softmax = bias_variable([196])

conv1_1 = conv(x, 3, 64)
conv1_2 = conv(conv1_1, 64, 64)
maxpool_1 = maxpool(conv1_2)

conv2_1 = conv(maxpool_1, 64, 128)
conv2_2 = conv(conv2_1, 128, 128)
maxpool_2 = maxpool(conv2_2)

conv3_1 = conv(maxpool_2, 128, 256)
conv3_2 = conv(conv3_1, 256, 256)
conv3_3 = conv(conv3_2, 256, 256)
maxpool_3 = maxpool(conv3_3)

conv4_1 = conv(maxpool_3, 256, 512)
conv4_2 = conv(conv4_1, 512, 512)
conv4_3 = conv(conv4_2, 512, 512)
maxpool_4 = maxpool(conv4_3)

conv5_1 = conv(maxpool_4, 512, 512)
conv5_2 = conv(conv5_1, 512, 512)
conv5_3 = conv(conv5_2, 512, 512)
maxpool_5 = maxpool(conv5_3)

x_reshape = tf.reshape(maxpool_5, [-1, 7 * 7 * 512])
fc_1 = tf.nn.relu(tf.matmul(x_reshape, W_fc1) + b_fc1)
fc_2 = tf.nn.relu(tf.matmul(fc_1, W_fc2) + b_fc2)
fc_3 = tf.nn.relu(tf.matmul(fc_2, W_fc3) + b_fc3)

h_fc_drop = tf.nn.dropout(fc_3, keep_prob)

y_conv = tf.nn.softmax(tf.matmul(fc_3, W_softmax) + b_softmax)

cross_entropy = -tf.reduce_sum(y * tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 2))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    for step in range(20000):
        data, label = d.batch(25)
        if step % 100 == 0:
            print('!!! Checkpoint Step = %d Created !!!' % step)
            saver.save(sess, r'checkpoint/ck1', global_step=step)
        if step % 50 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x: data, y: label, keep_prob: 1.0})
            print("step %d, training accuracy %g" % (step, train_accuracy))
        train_step.run(feed_dict={x: data, y: label, keep_prob: 0.5})
    final_saver = tf.train.Saver()
    final_saver.save(sess, r'/graph_save/save')
