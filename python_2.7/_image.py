import os

import scipy.io as sio
from PIL import Image

image_dir = r'F:/car_ims/'
output_dir = r'F:/image_output/'
matfn_1 = r'G:\car\car\matlab\cars_annos.mat'
matfn_2 = r'G:\car\car\matlab\cars_meta.mat'
car_data = sio.loadmat(matfn_1)
car_name = sio.loadmat(matfn_2)

# print(car_data['annotations'][0][2][5][0])
print(car_name['class_names'][0][1][0])


def work(image_name, x1, y1, x2, y2, class_name):
    im = Image.open(image_dir + image_name[7:])
    box = (x1, y1, x2, y2)
    region = im.crop(box)
    region = region.resize((256, 256))
    region.save(output_dir + car_name['class_names'][0][int(class_name) - 1][0] + image_name[7:])


def create_dir():
    for i in range(195):
        if not os.path.exists(output_dir + car_name['class_names'][0][i + 1][0]):
            os.makedirs(output_dir + car_name['class_names'][0][i + 1][0])


if __name__ == '__main__':
    create_dir()
    for i in range(16184):
        print (str(i/16184) + '%')
        work(car_data['annotations'][0][i][0][0],
             car_data['annotations'][0][i][1][0],
             car_data['annotations'][0][i][2][0],
             car_data['annotations'][0][i][3][0],
             car_data['annotations'][0][i][4][0],
             car_data['annotations'][0][i][5][0]
             )
