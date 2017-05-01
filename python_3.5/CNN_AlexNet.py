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
                         padding='SAME', name='conv2')
    conv2_relu = tf.nn.relu(conv2 + biases2, name='relu2')
    pool2 = tf.nn.max_pool(conv2_relu,
                           ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='maxpool2')
    norm2 = tf.nn.local_response_normalization(pool2, depth_radius=5, name='LRN2')

with tf.name_scope('conv3') as scope_2:
    weights3 = tf.Variable(tf.truncated_normal([3, 3, 256, 384]), name='weights3')
    biases3 = tf.Variable(tf.truncated_normal([384]), name='biases3')
    conv3 = tf.nn.conv2d(norm2,
                         filter=weights3, strides=[1, 1, 1, 1],
                         padding='SAME', name='conv3')
    conv3_relu = tf.nn.relu(conv3 + biases3, name='relu3')

with tf.name_scope('conv4') as scope_3:
    weights4 = tf.Variable(tf.truncated_normal([3, 3, 384, 384]), name='weights4')
    biases4 = tf.Variable(tf.truncated_normal([384]), name='biases4')
    conv4 = tf.nn.conv2d(conv3_relu,
                         filter=weights4, strides=[1, 1, 1, 1],
                         padding='SAME', name='conv4')
    conv4_relu = tf.nn.relu(conv4 + biases4, name='relu4')

with tf.name_scope('conv5') as scope_4:
    weights5 = tf.Variable(tf.truncated_normal([3, 3, 384, 256]), name='weights5')
    biases5 = tf.Variable(tf.truncated_normal([256]), name='biases5')
    conv5 = tf.nn.conv2d(conv4_relu,
                         filter=weights5, strides=[1, 1, 1, 1],
                         padding='SAME', name='conv5')
    conv5_relu = tf.nn.relu(conv5 + biases5, name='relu5')
    pool5 = tf.nn.max_pool(conv5_relu,
                           ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='pool5')

with tf.name_scope('fc1') as scope_5:
    x_reshape = tf.reshape(pool5, [-1, 6 * 6 * 256], name='reshape')
    weights6 = tf.Variable(tf.truncated_normal([6 * 6 * 256, 4096]), name='weights6')
    biases6 = tf.Variable(tf.truncated_normal([4096]), name='biases6')
    fc1 = tf.matmul(x_reshape, weights6, name='fc1')
    fc1_relu = tf.nn.relu(fc1 + biases6, name='relu6')
    drop_out1 = tf.nn.dropout(fc1_relu, keep_prob=keep_prob, name='drop_out1')

with tf.name_scope('fc2') as scope_6:
    weights7 = tf.Variable(tf.truncated_normal([4096, 4096]), name='weights7')
    biases7 = tf.Variable(tf.truncated_normal([4096]), name='biases7')
    fc2 = tf.matmul(drop_out1, weights7, name='fc2')
    fc2_relu = tf.nn.relu(fc2 + biases7, name='relu7')
    drop_out2 = tf.nn.dropout(fc2_relu, keep_prob=keep_prob, name='drop_out2')

with tf.name_scope('fc3') as scope_7:
    weights8 = tf.Variable(tf.truncated_normal([4096, 196]), name='weights8')
    biases8 = tf.Variable(tf.truncated_normal([196]), name='biases8')
    fc3 = tf.nn.bias_add(tf.matmul(drop_out2, weights8), biases8, name='fc3')

with tf.name_scope('loss') as scope_8:
    cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fc3, labels=y))
    tf.summary.scalar('cross_entropy', tf.reduce_mean(cross_entropy))
    train_step = tf.train.AdamOptimizer(1e-7).minimize(cross_entropy)

with tf.name_scope('train') as scope_9:
    correct_pred = tf.equal(tf.arg_max(fc3, 1), tf.arg_max(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    tic()
    sess.run(init)
    shutil.rmtree('/tmp/log/')
    saver = tf.train.Saver()
    merged_summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter('/tmp/log', sess.graph)
    for step in range(100000):
        data, label = d.batch(48)
        if step % 1000 == 0:
            print('!!! Checkpoint Step = %d Created !!!' % step)
            saver.save(sess, r'checkpoint/ck1', global_step=step)
        if step % 50 == 0:
            summary_str, train_accuracy = sess.run([merged_summary_op, accuracy], feed_dict={
                x: data, y: label, keep_prob: 1.0})
            print("step %d, training accuracy %g" % (step, train_accuracy))
            summary_writer.add_summary(summary_str, step)
        train_step.run(feed_dict={x: data, y: label, keep_prob: 0.5})
    toc()
    final_saver = tf.train.Saver()
    final_saver.save(sess, r'graph_save/save')
