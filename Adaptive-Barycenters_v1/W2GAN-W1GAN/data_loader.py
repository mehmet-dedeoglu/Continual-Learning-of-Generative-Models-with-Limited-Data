import torch.utils.data as data_utils
import torch
import numpy as np
import os
import glob
from PIL import Image
from torch.autograd import Variable
import shutil
from Converter import download_dataset
import networks
from DNN_MNIST import CNN_Gen_gp


def move_data(args, cls, samp_num, folder_name):
    class_ = args.train_classes[cls]
    # Check if dataset exists
    data_folder = args.dataset + '_data'
    if not os.path.exists(data_folder):
        download_dataset(data_folder)
    sim_dest = args.dataset + '_Simulations'
    if not os.path.exists(sim_dest):
        os.mkdir(sim_dest)
    dest_folder_name = folder_name
    if not os.path.exists(dest_folder_name):
        os.mkdir(dest_folder_name)
    dest_fid_folder_name = folder_name + '_fid'
    if not os.path.exists(dest_fid_folder_name):
        os.mkdir(dest_fid_folder_name)
    if type(class_) is not str:
        # Move data to the destination folder.
        fps_train_temp = [args.data_folder + '/' + task_ + '/Train' for task_ in os.listdir(args.data_folder)
                          if int(os.path.split(task_)[-1].split('_')[1]) in [class_]]
        for ell in range(len(fps_train_temp)):
            files = glob.glob(fps_train_temp[ell] + '/*')
            file_idx = np.arange(len(files))
            fid_file_idx = np.arange(len(files))
            for k in fid_file_idx:
                shutil.copy2(files[k], dest_fid_folder_name)
            if samp_num == 'All':
                files_copy_idx = file_idx
            else:
                permuted_idx = np.random.permutation(file_idx)
                files_copy_idx = permuted_idx[:samp_num]
            for k in files_copy_idx:
                shutil.copy2(files[k], dest_folder_name)
    else:
        model_struct = CNN_Gen_gp(args.channel_number)
        checkpoint = torch.load(class_)
        model_struct.load_state_dict(checkpoint['modelG'])
        model_struct.eval()
        seed = torch.randn((samp_num, args.noise_dim[0], args.noise_dim[1], args.noise_dim[2]))
        seed = get_torch_variable(args, seed)
        # For RAM efficiency and limitations, generate images in splits
        im_index = 0
        split_number = 10
        for split in range(split_number):
            generated_image = model_struct(seed[int(split*samp_num/split_number):int((split+1)*samp_num/split_number)])
            generated_image = generated_image.mul(0.5).add(0.5).data.cpu()
            generated_image = generated_image * 255
            data = np.asarray(generated_image, dtype="uint8")
            gen_folder = dest_folder_name
            if not os.path.exists(gen_folder):
                os.mkdir(gen_folder)
            for m in range(int(samp_num/split_number)):
                image_name = gen_folder + '/Label_' + str(cls) + '_Train_' + str(im_index) + '_.png'
                im = Image.fromarray(data[m, 0, :, :])
                im.save(image_name)
                im_index += 1
            del generated_image, data, im
            torch.cuda.empty_cache()

    return dest_folder_name


def move_data2(args, cls, samp_num, folder_name):
    class_ = args.train_classes[cls]
    # Check if dataset exists
    data_folder = args.dataset + '_data'
    if not os.path.exists(data_folder):
        download_dataset(data_folder)
    sim_dest = args.dataset + '_Simulations'
    if not os.path.exists(sim_dest):
        os.mkdir(sim_dest)
    dest_folder_name = folder_name
    if not os.path.exists(dest_folder_name):
        os.mkdir(dest_folder_name)
    dest_fid_folder_name = folder_name + '_fid'
    if not os.path.exists(dest_fid_folder_name):
        os.mkdir(dest_fid_folder_name)
    if type(class_) is not str:
        # Move data to the destination folder.
        fps_train_temp = [args.data_folder + '/' + task_ + '/Train' for task_ in os.listdir(args.data_folder)
                          if int(os.path.split(task_)[-1].split('_')[1]) in [class_]]
        for ell in range(len(fps_train_temp)):
            files = glob.glob(fps_train_temp[ell] + '/*')
            file_idx = np.arange(len(files))
            fid_file_idx = np.arange(len(files))
            for k in fid_file_idx:
                shutil.copy2(files[k], dest_fid_folder_name)
            if samp_num == 'All':
                files_copy_idx = file_idx
            else:
                permuted_idx = np.random.permutation(file_idx)
                files_copy_idx = permuted_idx[:samp_num]
            for k in files_copy_idx:
                shutil.copy2(files[k], dest_folder_name)
    else:
        model_struct = networks.get_g(args.config)
        checkpoint = torch.load(class_)
        model_struct.load_state_dict(checkpoint['modelG'])
        model_struct.eval()
        seed = torch.randn((samp_num, args.noise_dim[0], args.noise_dim[1], args.noise_dim[2]))
        seed = Variable(seed).cuda(args.cuda)
        # seed = get_torch_variable(args, seed)
        # For RAM efficiency and limitations, generate images in splits
        im_index = 0
        split_number = 10
        template = torch.randn((int(samp_num/split_number), 1, 32, 32))
        for split in range(split_number):
            generated_image = model_struct(seed[int(split*samp_num/split_number):int((split+1)*samp_num/split_number)],
                                           template)
            generated_image = generated_image.mul(0.5).add(0.5).data.cpu()
            generated_image = generated_image * 255
            data = np.asarray(generated_image, dtype="uint8")
            gen_folder = dest_folder_name
            if not os.path.exists(gen_folder):
                os.mkdir(gen_folder)
            for m in range(int(samp_num/split_number)):
                image_name = gen_folder + '/Label_' + str(cls) + '_Train_' + str(im_index) + '_.png'
                im = Image.fromarray(data[m, 0, :, :])
                im.save(image_name)
                im_index += 1
            del generated_image, data, im
            torch.cuda.empty_cache()

    return dest_folder_name


def get_infinite_batches(data_loader):
    while True:
        for i, (images) in enumerate(data_loader):
            yield images


def get_torch_variable(args, val):
    if args.cuda:
        return Variable(val).cuda(0)
    else:
        return Variable(val)


def get_data_loader(args, dest):
    files = glob.glob(dest + '/*')
    new = []
    for k in range(len(files)):
        data = np.asarray(Image.open(files[k]), dtype="float32")
        if args.dataset == 'MNIST':
            data = data.reshape(1, 32, 32)
        else:
            data = np.transpose(data, [2, 0, 1])
        data = (data - 127.5) / 127.5  # Normalize the images to [-1, 1]
        new.append(data)
        print(str(k))
    dataset_train = np.array(new)
    # Check if batch size is larger than total dataset size! To be completed...
    data = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True,
                                       num_workers=args.workers, pin_memory=True)
    return data


def get_data_loader_custom(args, size, train_classes, subsample=0):
    if type(train_classes[0]) is not list:
        train_classes = [train_classes]
    dataset_train = []
    for _a in range(len(train_classes)):
        fps_train_temp = [args.data_folder + '/' + task_ + '/Train' for task_ in os.listdir(args.data_folder)
                          if int(os.path.split(task_)[-1].split('_')[1]) in train_classes[_a]]
        dataset_hor = []
        for ell in range(len(train_classes[_a])):
            files = glob.glob(fps_train_temp[ell] + '/*')
            new = []
            for k in range(len(files)):
                data = np.asarray(Image.open(files[k]), dtype="float32")
                data = data.reshape(1, 32, 32)
                data = (data - 127.5) / 127.5  # Normalize the images to [-1, 1]
                new.append(data)
                print(str(k))
            new = np.array(new)
            dataset_hor.extend(new)
        dataset_train.append(np.array(dataset_hor))

    total_sample_num = dataset_train[0].shape[0]
    if subsample != 0:  # Can only subsample from a single item list 'dataset_train'.
        random_indices = np.asarray(np.floor(np.random.uniform(0, total_sample_num,
                                                               subsample)),  dtype='uint16')
        dataset_train[0] = dataset_train[0][random_indices]
    # Create a dataset
    train_ds_ = []
    test_ds_ = []
    for i in range(len(dataset_train)):
        train_ds_.append(torch.utils.data.DataLoader(dataset_train[i], batch_size=size, shuffle=True,
                                                     num_workers=args.workers, pin_memory=True))
    return train_ds_, test_ds_
