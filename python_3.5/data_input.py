import os
import numpy as np
import cv2
from sklearn.preprocessing import OneHotEncoder

class_name = os.listdir(r'/home/fpsandnoob/image_output/')
class_dict = {}
for i in range(len(class_name)):
    class_dict[class_name[i]] = i
class_array = np.arange(len(class_name))
enc = OneHotEncoder()
enc.fit([[i] for i in class_array])


def One_Hot(str):
    i = class_dict[str]
    return enc.transform(i).toarray()


class Data:
    def __init__(self):
        self.data_ = []
        self.label_ = []
        self.data = []
        self.label = []
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._num_examples = 0
        self.data_path_dir = os.listdir(r'/home/fpsandnoob/image_output/')

    def get_img(self):
        for dir_path in self.data_path_dir:
            p = os.listdir("/home/fpsandnoob/image_output/" + dir_path)
            for im in p:
                self.data.append(cv2.imread(os.path.join("/home/fpsandnoob/image_output/" + dir_path, im)))
                self.label.append(One_Hot(dir_path))
                self._num_examples += 1
        print("!!!!DATA LOAD Finished!!!!")

    def batch(self, size):
        start = self._index_in_epoch
        self._index_in_epoch += size
        if self._index_in_epoch > self._num_examples:
            self._epochs_completed += 1
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            for i in perm:
                self.data_.append(self.data[i])
                self.label_.append(self.label[i])
            start = 0
            self._index_in_epoch = size
            assert size <= self._num_examples
            end = self._index_in_epoch
            self.data = self.data_
            self.label = self.label_
            return self.data_[start:end], self.label_[start:end]
        end = self._index_in_epoch
        return self.data[start:end], self.label[start:end]

#
# d1 = Data()
# d1.get_img()
# for i in range(10):
#     data, label = d1.batch(10000)
# # print(np.shape(label))
