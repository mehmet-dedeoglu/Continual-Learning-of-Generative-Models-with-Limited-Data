from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from scipy.linalg import sqrtm
import numpy as np
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from skimage.transform import resize
from numpy import asarray
import os
from Plot_Figures import plot_from_txt
from PIL import Image
import tensorflow as tf
from DNN_FID import Score_Model
from config import parse_args_fid


def scale_images(images, new_shape):
    images_list = list()
    for image in images:
        # resize with nearest neighbor interpolation
        new_image = resize(image, new_shape, 0)
        images_list.append(new_image)
    return asarray(images_list)


def calculate_fid_score2(imf, imr, classifier_model):

    images_fake = imf.reshape(imf.shape[0], imf.shape[1], imf.shape[2], 1)
    images_real = imr.reshape(imr.shape[0], imr.shape[1], imr.shape[2], 1)
    images_fake = images_fake[:, :, :, :] / 255.0
    images_real = images_real[:, :, :, :] / 255.0
    act_fake = classifier_model.predict(images_fake)
    act_real = classifier_model.predict(images_real)
    mu_fake, sigma_fake = act_fake.mean(axis=0), cov(act_fake, rowvar=False)
    mu_real, sigma_real = act_real.mean(axis=0), cov(act_real, rowvar=False)
    ssdiff = np.sum((mu_fake - mu_real) ** 2.0)
    covmean = sqrtm(sigma_fake.dot(sigma_real))
    if iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + trace(sigma_fake + sigma_real - 2.0 * covmean)
    print('FID: ', fid)
    return fid


def train_model(classifier_model, checkpoint, checkpoint_prefix):
    # Load MNIST dataset
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = 255 * scale_images(x_train, (32, 32, 1))
    x_test = 255 * scale_images(x_test, (32, 32, 1))

    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    # Train model once
    classifier_model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
    classifier_model.summary()
    classifier_model.fit(x_train, y_train, batch_size=128, epochs=10)
    classifier_model.evaluate(x_test, y_test, verbose=2)
    # Save checkpoint to be used later without training
    checkpoint.save(file_prefix=checkpoint_prefix)
    return classifier_model


# assumes images have any shape 299*299*3 and pixel values in [0,255]
def calculate_fid_score(imf, imr, classifier_model):

    # Convert 2D images to 3D RGB images
    images_fake = []
    images_real = []
    for _ in range(3):
        images_fake.append(imf)
        images_real.append(imr)
    images_fake = np.asarray(images_fake)
    images_real = np.asarray(images_real)
    images_fake = np.transpose(images_fake, [1, 2, 3, 0])
    images_real = np.transpose(images_real, [1, 2, 3, 0])
    images_fake = 255 * scale_images(images_fake, (299, 299, 3))
    images_real = 255 * scale_images(images_real, (299, 299, 3))
    images_fake = preprocess_input(images_fake)
    images_real = preprocess_input(images_real)

    act_real = classifier_model.predict(images_real)
    act_fake = classifier_model.predict(images_fake)
    mu_fake, sigma_fake = act_fake.mean(axis=0), cov(act_fake, rowvar=False)
    mu_real, sigma_real = act_real.mean(axis=0), cov(act_real, rowvar=False)
    ssdiff = np.sum((mu_fake - mu_real) ** 2.0)
    covmean = sqrtm(sigma_fake.dot(sigma_real))
    if iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + trace(sigma_fake + sigma_real - 2.0 * covmean)
    print('FID: ', fid)
    return fid


def read_from_folder(path, index):
    dataset = []
    for i in range(len(path)):
        files = os.listdir(path[i])
        new = []
        for k in range(len(index)):
            data = np.asarray(Image.open(path[i] + '/' + str(files[index[k]])), dtype="uint8")
            new.append(data)
        new = np.array(new)
        dataset.append(new)
    return dataset


if __name__ == '__main__':
    args = parse_args_fid()

    num_of_samples = args.sample_number
    folder_path = args.folder_path
    string_fid = os.path.split(folder_path)[0] + 'FID_score'
    string_iters = os.path.split(folder_path)[0] + 'FID_score_iters'
    # Check if FID document already generated
    if not os.path.exists(string_fid):
        # load classifier model
        if args.select == 'modified_fid_score':
            _model = Score_Model()
            checkpoint = tf.train.Checkpoint(_model=_model)
            checkpoint_dir = './MNIST_score_model'
            checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
            # If a pretrained model does not exist, create and train a model for future use.
            if not os.path.exists(checkpoint_dir):
                os.mkdir(checkpoint_dir)
                _model = train_model(_model, checkpoint, checkpoint_prefix)
            else:
                checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
            classifier_model_ = tf.keras.Sequential(_model.layers[0:-1])
        else:
            classifier_model_ = InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))

        iterations = [int(os.path.split(names)[-1].split('_')[1]) for names in os.listdir(folder_path)]
        iterations.sort()
        score = []
        range_array = []
        range_array.extend(range(0, len(iterations), 1))
        for it in range_array:
            fake_path = folder_path + '/Iteration_' + str(iterations[it]) + '/Fake_Images'
            # real_path = folder_path + '/Iteration_' + str(iterations[it]) + '/Real_Images'
            real_path = 'MNIST_Simulations/Folder_EdgeOnly_[8, 9]Full13/FID_Images/Iteration_' + str(iterations[it]) + '/Real_Images'
            max_index = len(os.listdir(fake_path))
            current_index = 0
            mean_score = 0
            k = 0
            while current_index < max_index - 1:
                indices = range(current_index, min(current_index + num_of_samples, max_index))
                datasets = read_from_folder([fake_path, real_path], indices)
                if args.select == 'modified_fid_score':
                    mean_score = mean_score + calculate_fid_score2(datasets[0], datasets[1], classifier_model_)
                else:
                    mean_score = mean_score + calculate_fid_score(datasets[0], datasets[1], classifier_model_)
                current_index = max(indices) + 1
                k += 1
            score.append(mean_score/k)
            print(it)
        score_save = np.array(score)
        iterations_save = np.array(iterations)
        np.savetxt(string_fid, score_save, delimiter=',')
        np.savetxt(string_iters, iterations_save, delimiter=',')

    fig_labels = args.labels
    legend_labels = args.legend_text
    fig_text = [fig_labels[0], fig_labels[1], fig_labels[2], legend_labels,
                os.path.split(folder_path)[0] + 'Score.svg', os.path.split(folder_path)[0] + 'Score.png']
    plot_from_txt([string_fid], [string_iters], fig_text)

    print('Finished')
