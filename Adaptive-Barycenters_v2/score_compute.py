from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from scipy.linalg import sqrtm
import numpy as np
# from keras.applications.inception_v3 import InceptionV3
# from keras.applications.inception_v3 import preprocess_input
from skimage.transform import resize
from numpy import asarray
import os
from PIL import Image
import tensorflow as tf
from archs.wb_networks import Score_Model
import shutil
import glob
import metrics.inception_score_tf1 as inception_score_tf1
import metrics.fid_score_tf1 as fid_score_tf1
from utils.config import parse_args
from utils.converter import download_dataset
from metrics.geom_score import geom_score, rlts
from metrics.MM_dicrepancy import MMD

# import pylab as plt


def move_data(args, cls, samp_num, folder_name, cls_=[]):
    if type(cls) is int:
        class_ = args.train_classes[cls]
    else:
        class_ = cls_
        cls = 'generated'
    # Check if dataset exists
    data_folder = args.dataset_prefix + args.dataset + '_data'
    if not os.path.exists(data_folder):
        download_dataset(data_folder)
    sim_dest = args.dataset_prefix + args.dataset + '_Simulations'
    if not os.path.exists(sim_dest):
        os.mkdir(sim_dest)
    dest_folder_name = folder_name
    if not os.path.exists(dest_folder_name):
        os.mkdir(dest_folder_name)
    if type(class_) is not str:
        # Move data to the destination folder.
        fps_train_temp = [data_folder + '/' + task_ + '/Train' for task_ in os.listdir(data_folder)
                          if int(os.path.split(task_)[-1].split('_')[1]) in [class_]]
        for ell in range(len(fps_train_temp)):
            files = glob.glob(fps_train_temp[ell] + '/*')
            file_idx = np.arange(len(files))
            if samp_num == 'All':
                files_copy_idx = file_idx
            else:
                permuted_idx = np.random.permutation(file_idx)
                files_copy_idx = permuted_idx[:samp_num]
            for k in files_copy_idx:
                shutil.copy2(files[k], dest_folder_name)


def scale_images(images, new_shape):
    images_list = list()
    # k = 1
    for image in images:
        # resize with nearest neighbor interpolation
        # print('Resizing image number ' + str(k))
        # k += 1
        new_image = resize(image, new_shape, 0)
        images_list.append(new_image)
    return asarray(images_list)


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


def read_from_folder(path, index):
    dataset = []
    for i in range(len(path)):
        files = os.listdir(path[i])
        # print(files)
        new = []
        for k in range(len(index)):
            # print(str(k) + ' path ' + path[i] + '/' + str(files[index[k]]))
            data = np.asarray(Image.open(path[i] + '/' + str(files[index[k]])), dtype="uint8")
            new.append(data)
        new = np.array(new)
        dataset.append(new)
    return dataset


def calculate_MFID(imf, imr, classifier_model):

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


# assumes images have any shape 299*299*3 and pixel values in [0,255]
def calculate_IS(imf, data='MNIST'):
    if data == 'MNIST':
        # Convert 2D images to 3D RGB images
        images_fake = []
        for _ in range(3):
            images_fake.append(imf)
        images_fake = np.asarray(images_fake)
        images_fake = np.transpose(images_fake, [1, 0, 2, 3])
    else:
        images_fake = imf
        images_fake = np.transpose(images_fake, [0, 3, 1, 2])
    result_dict = inception_score_tf1.get_inception_score(images_fake, splits=10)
    print('Mean IS is : ' + str(result_dict[0]))
    return result_dict[0], result_dict[1]


# assumes images have any shape 299*299*3 and pixel values in [0,255]
def calculate_FID(imf, imr, data='MNIST'):
    if data == 'MNIST':
        # Convert 2D images to 3D RGB images
        images_fake = []
        images_real = []
        for _ in range(3):
            images_fake.append(imf)
            images_real.append(imr)
        images_fake = np.asarray(images_fake)
        images_real = np.asarray(images_real)
        images_fake = np.transpose(images_fake, [1, 0, 2, 3])
        images_real = np.transpose(images_real, [1, 0, 2, 3])
    else:
        images_fake = imf
        images_fake = np.transpose(images_fake, [0, 3, 1, 2])
        images_real = imr
        images_real = np.transpose(images_real, [0, 3, 1, 2])
    fid = fid_score_tf1.get_fid(images_real, images_fake)
    print('FID: ', fid)
    return fid


def calculate_GS(imf, imr, data='MNIST'):
    if data == 'MNIST':
        fake_image = np.reshape(imf, (-1, 784))
        real_image = np.reshape(imf, (-1, 784))
    elif data == 'CIFAR10':
        fake_image = np.reshape(imf, (-1, 32*32*3))
        real_image = np.reshape(imf, (-1, 32*32*3))
    rlts_fake = rlts(fake_image, L_0=64, gamma=None, i_max=100, n=len(imf))
    rlts_real = rlts(real_image, L_0=64, gamma=None, i_max=100, n=len(imr))
    gs = geom_score(rlts_fake, rlts_real)
    print('GS score is: ', gs)
    return gs


def calculate_MMD(imf, imr, data='MNIST'):
    if data == 'MNIST':
        fake_image = np.reshape(imf, (-1, 784))
        real_image = np.reshape(imr, (-1, 784))
    elif data == 'CIFAR10':
        fake_image = np.reshape(imf, (-1, 32*32*3))
        real_image = np.reshape(imr, (-1, 32*32*3))
    MMDS = MMD(fake_image, real_image, kernel='multiscale')
    print('MMD score is: ', MMDS)
    return MMDS


def score_manager(arg, instr):
    location = arg.train_classes[instr[1][0][0]]
    arg.dataset = location.split("/")[-4].split('_')[0]
    score_path = location.split('checkpoints')[0] + 'FID'
    check_path = location.split('checkpoints')[0] + 'checkpoints'
    checkpoints = [[check_path + '/' + x, int(x.split('_')[-1])] for x in os.listdir(check_path)
                   if x.split('_')[0] == 'checkpoint']
    checkpoints.sort(key=lambda i: i[1])

    folder_name = os.listdir(score_path)
    # print(folder_name)
    numOfClasses = int(len(instr[0][0]) / 2)
    if 'Data_folder_dataset' not in folder_name:
        # print('Data_folder_dataset is not in folder_name.')
        folder_name.append('Data_folder_dataset')
    else:
        shutil.rmtree(score_path + '/Data_folder_dataset')
    folder_name = [score_path + '/' + pth for pth in folder_name]
    ref_data_name = [names for names in folder_name if names.split('_')[-1] == 'dataset'][0]
    dataset_sample = 0
    for class_index in range(numOfClasses):
        dataset_index = instr[0][0][2 * class_index]
        sample_size = instr[0][0][2 * class_index + 1]
        dataset_sample += sample_size
        if arg.dnn_type == 1:
            move_data(arg, dataset_index, sample_size, ref_data_name)
    print('Datasets are ready for score computation.')

    if arg.score_type == 'MFID':
        _mdl = Score_Model()
        chkpnt = tf.train.Checkpoint(_model=_mdl)
        chkpnt_dir = './Datasets/MNIST_score_model'
        chkpnt_prefix = os.path.join(chkpnt_dir, "ckpt")
        # If a pretrained model does not exist, create and train a model for future use.
        if not os.path.exists(chkpnt_dir):
            os.mkdir(chkpnt_dir)
            _mdl = train_model(_mdl, chkpnt, chkpnt_prefix)
        else:
            chkpnt.restore(tf.train.latest_checkpoint(chkpnt_dir))
        cls_model_ = tf.keras.Sequential(_mdl.layers[0:-1])
    elif arg.score_type == 'GS' or arg.score_type == 'MMD':
        pass
    else:
        print('No need to download inception model in this implementation.')
        # print('Loading inception model.')
        # cls_model_ = InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3),
        #                          weights='imagenet', classes=1000)

    # elif arg.score_type == 'IS':
    #     string_fid_ = score_path + '_IS_score'
    #     string_iters_ = score_path + '_IS_score_iters'
    # elif arg.score_type == 'MFID':
    #     string_fid_ = score_path + '_MFID_score'
    #     string_iters_ = score_path + '_MFID_score_iters'
    fld_name = [[x, int(x.split('_')[-1])] for x in folder_name if x.split('_')[-1] != 'dataset']
    fld_name.sort(key=lambda i: i[1])
    # print(fld_name)
    fld_name = [x for x in fld_name if x[1] % arg.checkpoint_freq == 0]
    folder_name = [x[0] for x in fld_name]
    iters = [x[1] for x in fld_name]
    # print(fld_name)
    # iters = [int(names.split('_')[-1]) for names in folder_name if names.split('_')[-1] != 'dataset']
    # iters.sort()

    score_ = []
    score_std_ = []
    string_std_ = score_path + '_score_std'
    string_fid_ = score_path + '_score'
    string_iters_ = score_path + '_score_iters'
    range_array_ = []
    range_array_.extend(range(0, len(iters), 1))
    max_idx = dataset_sample
    num_of_samples_ = arg.score_split_size

    for i_ in range_array_:
        # print('Entered for loop')
        fake_path_ = folder_name[i_]
        real_path_ = ref_data_name
        cnt_index = 0
        mean_score_ = 0
        std_score_ = 0
        kk = 0
        while cnt_index < max_idx - 1:
            idxs = range(cnt_index, min(cnt_index + num_of_samples_, max_idx))
            # print(fake_path_)
            # print(real_path_)
            # print(idxs)
            datasets_ = read_from_folder([fake_path_, real_path_], idxs)
            if arg.score_type == 'MFID':
                mean_score_ += calculate_MFID(datasets_[0], datasets_[1], cls_model_)
                string_fid_ = score_path + '_MFID_score'
                string_iters_ = score_path + '_MFID_score_iters'
            elif arg.score_type == 'IS':
                string_fid_ = score_path + '_IS_score'
                string_iters_ = score_path + '_IS_score_iters'
                string_std_ = score_path + '_IS_score_std'
                # print('Entered IS:')
                mean_score_new, std_score_new = calculate_IS(datasets_[0], data=arg.dataset)
                mean_score_, std_score_ = mean_score_ + mean_score_new, std_score_ + std_score_new
                # mean_score_ += inception_score.inception_score(datasets_[0], cuda=True, batch_size=10,
                #                                                resize=True, splits=10)
            elif arg.score_type == 'GS':
                string_fid_ = score_path + '_GS_score'
                string_iters_ = score_path + '_GS_score_iters'
                mean_score_ += calculate_GS(datasets_[0], datasets_[1], data=arg.dataset)
            elif arg.score_type == 'MMD':
                string_fid_ = score_path + '_MMD_score'
                string_iters_ = score_path + '_MMD_score_iters'
                mean_score_ += calculate_MMD(datasets_[0], datasets_[1], data=arg.dataset)
            else:
                string_fid_ = score_path + '_FID_score'
                string_iters_ = score_path + '_FID_score_iters'
                mean_score_ += calculate_FID(datasets_[0], datasets_[1], data=arg.dataset)
                # mean_score_ += fid_score.main(arg, fake_path_, real_path_)
            cnt_index = max(idxs) + 1
            print('Checkpoint: ' + str(i_) + ' Split: ' + str(kk))
            kk += 1
        score_.append(mean_score_ / kk)
        score_std_.append(std_score_ / kk)
        print(i_)
    score_save_ = np.array(score_)
    iterations_save_ = np.array(iters)
    np.savetxt(string_fid_, score_save_, delimiter=',')
    np.savetxt(string_std_, score_std_, delimiter=',')
    np.savetxt(string_iters_, iterations_save_, delimiter=',')
    print('Computing the score is completed.')


def main(args):
    k = 0
    instructions = args.instructions
    while k < len(instructions):
        if instructions[k][0][0][0] == 'cuda':
            args.cuda = instructions[k][1][0][0]
        elif instructions[k][0][0][0] == 'batch_size':
            args.batch_size = instructions[k][1][0][0]
        elif instructions[k][0][0][0] == 'generator_iters':
            args.generator_iters = instructions[k][1][0][0]
        elif instructions[k][0][0][0] == 'critic_iter':
            args.critic_iter = instructions[k][1][0][0]
        elif instructions[k][0][0][0] == 'epoch_num':
            args.epoch_num = instructions[k][1][0][0]
        elif instructions[k][0][0][0] == 'beta_1':
            args.beta_1 = float(instructions[k][1][0][0])
        elif instructions[k][0][0][0] == 'beta_2':
            args.beta_2 = float(instructions[k][1][0][0])
        elif instructions[k][0][0][0] == 'epsilon':
            args.epsilon = float(instructions[k][1][0][0])
        elif instructions[k][0][0][0] == 'learning_rate':
            args.learning_rate = float(instructions[k][1][0][0])
        elif instructions[k][0][0][0] == 'lambda_val':
            args.lambda_val = instructions[k][1][0][0]
        elif instructions[k][0][0][0] == 'score_batch_size':
            args.score_batch_size = instructions[k][1][0][0]
        elif instructions[k][0][0][0] == 'dataset':
            args.dataset = instructions[k][1][0][0]
            if args.dataset == 'MNIST':
                args.channel_number = 1
                # args.data_folder = 'MNIST_data'
            elif args.dataset == 'CIFAR10':
                args.channel_number = 3
                # args.data_folder = 'CIFAR10_data'
            elif args.dataset == 'CIFAR100':
                args.channel_number = 3
                # args.data_folder = 'CIFAR100_data'
            elif args.dataset == 'LSUN':
                args.channel_number = 3
                # args.data_folder = 'LSUN_data'
        elif instructions[k][0][0][0] == 'subsample':
            args.subsample = instructions[k][1][0][0]
        elif instructions[k][0][0][0] == 'save_interval':
            args.save_interval = instructions[k][1][0][0]
        elif instructions[k][0][0][0] == 'FID_BATCH_SIZE':
            args.FID_BATCH_SIZE = instructions[k][1][0][0]
        elif instructions[k][0][0][0] == 'critic_weights':
            args.critic_weights = instructions[k][1][0]
        elif instructions[k][0][0][0] == 'precision':
            args.precision = instructions[k][1][0][0]
        elif instructions[k][0][0][0] == 'checkpoint_save':
            args.checkpoint_save = instructions[k][1][0][0]
        elif instructions[k][0][0][0] == 'mode':
            args.mode = instructions[k][1][0][0]
        elif instructions[k][0][0][0] == 'dnn_type':
            args.dnn_type = int(instructions[k][1][0][0])
        elif instructions[k][0][0][0] == 'score_split_size':
            args.score_split_size = int(instructions[k][1][0][0])
        elif instructions[k][0][0][0] == 'score_type':
            args.score_type = instructions[k][1][0][0]
        else:
            if args.mode == 'score':
                print('Entering score manager...')
                score_manager(args, instructions[k])
            else:
                print('Not implemented.')

        print(str(k) + 'th stage is completed.')
        k = k + 1


if __name__ == '__main__':
    arguments = parse_args()
    main(arguments)
    print('Finished')
