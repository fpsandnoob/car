from uitil import tic, toc
import tensorflow as tf
import data_input

d = data_input.Data()
d.get_img()
x = tf.placeholder('float', shape=[None, 28, 28, 3])
y = tf.placeholder('float', shape=[None, 196])
keep_prob = tf.placeholder('float')


def weight_variable(shape):
    initial = tf.truncated_normal(shape)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.truncated_normal(shape)
    return tf.Variable(initial)


def conv(x, _in, _out):
    return tf.nn.relu(tf.nn.conv2d(x, weight_variable([5, 5, _in, _out]), strides=[1, 1, 1, 1], padding="SAME") +
                      bias_variable([_out]))


def maxpool(x):
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")


conv_1 = conv(x, 3, 32)
pool_1 = maxpool(conv_1)

conv_2 = conv(pool_1, 32, 64)
pool_2 = maxpool(conv_2)

x_reshape = tf.reshape(pool_2, [-1, 7 * 7 * 64])
fc1 = tf.nn.relu(tf.matmul(x_reshape, weight_variable([7 * 7 * 64, 1024])) + bias_variable([1024]))

fc_1_drop = tf.nn.dropout(fc1, keep_prob)

y_conv = tf.nn.softmax(tf.matmul(fc_1_drop, weight_variable([1024, 196])) + bias_variable([196]))

cross_entropy = -tf.reduce_sum(y * tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.arg_max(y_conv, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

with tf.Session() as sess:
    tic()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    for step in range(20000):
        data, label = d.batch(25)
        if step % 1000 == 0:
            print('!!! Checkpoint Step = %d Created !!!' % step)
            saver.save(sess, r'checkpoint/ck1', global_step=step)
        if step % 50 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x: data, y: label, keep_prob: 1.0})
            print("step %d, training accuracy %g " % (step, train_accuracy))
        train_step.run(feed_dict={x: data, y: label, keep_prob: 0.5})
    toc()
    final_saver = tf.train.Saver()
    final_saver.save(sess, r'graph_save/save')
