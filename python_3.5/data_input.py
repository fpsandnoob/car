import os
import tensorflow as tf
import numpy as np

class_name = os.listdir(r'F:\image_output')
class_dict = {}
for i in range(len(class_name)):
    class_dict[class_name[i]] = i

def One_Hot(str):
    i = class_dict[str]
    return tf.one_hot(i, 196)


class Data:
    def __init__(self):
        self.data = []
        self.label = []
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._num_examples = 0
        self.data_path_dir = os.listdir(r'F:\image_output')

    def get_img(self):
        for dir_path in self.data_path_dir:
            p = os.listdir("F:/image_output/" + dir_path)
            for im in p:
                self.data.append(tf.reshape(tf.image.convert_image_dtype(tf.image.decode_jpeg(
                    tf.read_file(os.path.join("F:/image_output/" + dir_path, im)), channels=3),
                    dtype=tf.uint8), [224, 224, 3]))
                self.label.append([One_Hot(dir_path)])
                self._num_examples += 1

    def batch(self, size):
        start = self._index_in_epoch
        self._index_in_epoch += size
        if self._index_in_epoch > self._num_examples:
            self._epochs_completed += 1
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self.data = self.data[perm]
            self.label = self.label[perm]
            start = 0
            self._index_in_epoch = size
            assert size <= self._num_examples
        end = self._index_in_epoch
        return self.data[start:end], self.label[start:end]

d1 = Data()
d1.get_img()
d1.batch(10)
