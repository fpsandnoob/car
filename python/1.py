import scipy.io as sio

matfn_1 = r'G:\car\car\matlab\cars_annos.mat'
matfn_2 = r'G:\car\car\matlab\cars_meta.mat'
car_data = sio.loadmat(matfn_1)
car_name = sio.loadmat(matfn_2)
# print(car_data['annotations'][0][1][0][0])
print(car_name['class_names'][0][1])
