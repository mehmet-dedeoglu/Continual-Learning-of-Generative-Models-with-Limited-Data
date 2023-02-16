import os
import numpy as np
from skimage.transform import resize
from PIL import Image
import tensorflow as tf
from numpy import asarray

'''Splits MNIST dataset into distinct subfolders according to their classes.'''


def scale_images(images, new_shape):
    images_list = list()
    for image in images:
        # resize with nearest neighbor interpolation
        new_image = resize(image, new_shape, 0)
        # store
        images_list.append(new_image)
    return asarray(images_list)


def download_dataset(directory='MNIST_data'):
    if not os.path.exists(directory):
        os.mkdir(directory)
    all_directory = directory + '_all'
    if not os.path.exists(all_directory):
        os.mkdir(all_directory)
    if directory == 'MNIST_data':
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        print('Scaling images... This may take a moment...')
        x_train = 255 * scale_images(x_train, (32, 32))
        x_train = x_train.astype(np.uint8)
        x_test = 255 * scale_images(x_test, (32, 32))
        x_test = x_test.astype(np.uint8)
        print('Images are scaled.')
        for all in range(len(x_train)):
            image_name = all_directory + '/Label_' + str(all) + '_.png'
            im = Image.fromarray(x_train[all, :, :])
            im.save(image_name)
        for i in range(np.max(y_train)+1):
            sub_direc = directory + '/Class_' + str(i)
            if not os.path.exists(sub_direc):
                os.mkdir(sub_direc)

            train_ind = y_train[:] == i
            train_index = np.arange(len(train_ind), dtype=int)[train_ind]
            sub_direc_train = sub_direc + '/Train'
            if not os.path.exists(sub_direc_train):
                os.mkdir(sub_direc_train)
            for j in train_index:
                image_name = sub_direc_train + '/Label_' + str(i) + '_Train_' + str(j) + '_.png'
                im = Image.fromarray(x_train[j, :, :])
                im.save(image_name)
                print(str(j) + 'th training image in '+str(i)+'th class is saved.')

            test_ind = y_test[:] == i
            test_index = np.arange(len(test_ind), dtype=int)[test_ind]
            sub_direc_test = sub_direc + '/Test'
            if not os.path.exists(sub_direc_test):
                os.mkdir(sub_direc_test)
            for j in test_index:
                image_name = sub_direc_test + '/Label_' + str(i) + '_Test_' + str(j) + '_.png'
                im = Image.fromarray(x_test[j, :, :])
                im.save(image_name)
                print(str(j) + 'th testing image in '+str(i)+'th class is saved.')


if __name__ == '__main__':
    download_dataset()
