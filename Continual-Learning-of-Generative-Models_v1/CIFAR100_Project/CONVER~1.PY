import os
import numpy as np
from PIL import Image
import tensorflow as tf

'''Splits Cifar10 dataset into distinct subfolders according to their classes.'''


def MNIST_split():
    simulation_dir = 'CIFAR_Simulations'
    if not os.path.exists(simulation_dir):
        os.mkdir(simulation_dir)

    all_directory = 'CIFAR100_all_data'
    if not os.path.exists(all_directory):
        os.mkdir(all_directory)
    directory = 'CIFAR100_data'
    if not os.path.exists(directory):
        os.mkdir(directory)
    cifar10 = tf.keras.datasets.cifar100
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    for all in range(len(x_train)):
        image_name = all_directory + '/Label_' + str(all) + '_.png'
        im = Image.fromarray(x_train[all, :, :])
        im.save(image_name)
    for i in range(np.max(y_train)+1):
        sub_direc = directory + '/Class_' + str(i)
        if not os.path.exists(sub_direc):
            os.mkdir(sub_direc)

        train_ind = y_train[:, 0] == i
        train_index = np.arange(len(train_ind), dtype=int)[train_ind]
        sub_direc_train = sub_direc + '/Train'
        if not os.path.exists(sub_direc_train):
            os.mkdir(sub_direc_train)
        for j in train_index:
            image_name = sub_direc_train + '/Label_' + str(i) + '_Train_' + str(j) + '_.png'
            im = Image.fromarray(x_train[j, :, :])
            im.save(image_name)
            print(str(j) + 'th training image in ' + str(i) + 'th class is saved.')

        test_ind = y_test[:, 0] == i
        test_index = np.arange(len(test_ind), dtype=int)[test_ind]
        sub_direc_test = sub_direc + '/Test'
        if not os.path.exists(sub_direc_test):
            os.mkdir(sub_direc_test)
        for j in test_index:
            image_name = sub_direc_test + '/Label_' + str(i) + '_Test_' + str(j) + '_.png'
            im = Image.fromarray(x_test[j, :, :])
            im.save(image_name)
            print(str(j) + 'th testing image in ' + str(i) + 'th class is saved.')


def MNIST_split_samples():
    simulation_dir = 'CIFAR_Simulations'
    if not os.path.exists(simulation_dir):
        os.mkdir(simulation_dir)

    # all_directory = 'CIFAR100_all_data'
    # if not os.path.exists(all_directory):
    #     os.mkdir(all_directory)
    directory = 'CIFAR100_Sample'
    if not os.path.exists(directory):
        os.mkdir(directory)
    cifar10 = tf.keras.datasets.cifar100
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    # for all in range(len(x_train)):
    #     image_name = all_directory + '/Label_' + str(all) + '_.png'
    #     im = Image.fromarray(x_train[all, :, :])
    #     im.save(image_name)
    num_of_samples = len(y_train)
    sample_idx = np.arange(len(y_train))
    permuted_idx = np.random.permutation(sample_idx)
    border = [0, int(num_of_samples/5), int(num_of_samples/2), num_of_samples]
    for i in range(3):
        sub_direc = directory + '/Class_' + str(i)
        if not os.path.exists(sub_direc):
            os.mkdir(sub_direc)

        idx = permuted_idx[border[i]:border[i+1]]

        # train_ind = y_train[:, 0] == i
        # train_index = np.arange(len(train_ind), dtype=int)[train_ind]
        sub_direc_train = sub_direc + '/Train'
        if not os.path.exists(sub_direc_train):
            os.mkdir(sub_direc_train)
        for j in idx:
            image_name = sub_direc_train + '/Label_' + str(i) + '_Train_' + str(j) + '_.png'
            im = Image.fromarray(x_train[j, :, :])
            im.save(image_name)
            print(str(j) + 'th training image in ' + str(i) + 'th class is saved.')

        # test_ind = y_test[:, 0] == i
        # test_index = np.arange(len(test_ind), dtype=int)[test_ind]
        # sub_direc_test = sub_direc + '/Test'
        # if not os.path.exists(sub_direc_test):
        #     os.mkdir(sub_direc_test)
        # for j in test_index:
        #     image_name = sub_direc_test + '/Label_' + str(i) + '_Test_' + str(j) + '_.png'
        #     im = Image.fromarray(x_test[j, :, :])
        #     im.save(image_name)
        #     print(str(j) + 'th testing image in ' + str(i) + 'th class is saved.')


if __name__ == '__main__':
    MNIST_split()
