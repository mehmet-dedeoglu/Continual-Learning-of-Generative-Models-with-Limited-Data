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
from config import parse_args_fid


def scale_images(images, new_shape):
    images_list = list()
    for image in images:
        new_image = resize(image, new_shape, 0)
        images_list.append(new_image)
    return asarray(images_list)


# assumes images have any shape 299*299*3 and pixel values in [0,255]
def calculate_fid_score(images_fake, images_real, classifier_model):

    # images_real = 127.5*images_real + 127.5
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
        classifier_model_ = InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))

        iterations = [int(os.path.split(names)[-1].split('_')[1]) for names in os.listdir(folder_path)]
        iterations.sort()
        score = []
        range_array = []
        range_array.extend(range(0, len(iterations), 1))
        for it in range_array:
            fake_path = folder_path + '/Iteration_' + str(iterations[it]) + '/Fake_Images'
            real_path = folder_path + '/Iteration_' + str(iterations[it]) + '/Real_Images'
            max_index = len(os.listdir(fake_path))
            current_index = 0
            mean_score = 0
            k = 0
            while current_index < max_index - 1:
                indices = range(current_index, min(current_index + num_of_samples, max_index))
                datasets = read_from_folder([fake_path, real_path], indices)
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
