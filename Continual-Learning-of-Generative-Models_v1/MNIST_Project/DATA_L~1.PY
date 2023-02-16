import torch.utils.data as data_utils
import torch
import numpy as np
import os
import glob
from PIL import Image
from torch.autograd import Variable
import shutil


def get_infinite_batches(data_loader):
    while True:
        for i, (images) in enumerate(data_loader):
            yield images


def get_torch_variable(args, val):
    if args.cuda:
        return Variable(val).cuda(0)
    else:
        return Variable(val)


def ensemble(args, instructions):
    dest = 'MNIST_Simulations/ensemble_data_folder' + str(instructions[3])
    if not os.path.exists(dest):
        os.mkdir(dest)
    for ind in range(len(instructions[2])):
        path = args.train_classes[instructions[2][ind]]
        if type(path) is str:
            checkpoint = torch.load(path)
            G = checkpoint['modelG']
            G.eval()
            mm = 0
            samp_num = args.ensemble_sample_number  # For MNIST
            seed = torch.randn((samp_num, args.noise_dim[0], args.noise_dim[1], args.noise_dim[2]))
            seed = get_torch_variable(args, seed)
            for split in range(4):
                generated_image = G(seed[int(split * samp_num / 4):int((split + 1) * samp_num / 4)])
                generated_image = generated_image.mul(0.5).add(0.5).data.cpu()
                generated_image = generated_image * 255
                data = np.asarray(generated_image, dtype="uint8")
                gen_folder = dest
                for m in range(int(samp_num / 4)):
                    image_name = gen_folder + '/Image' + str(ind) + '_' + str(mm) + '_.png'
                    im = Image.fromarray(data[m, 0, :, :])
                    im.save(image_name)
                    mm += 1
                del generated_image, data, im
                torch.cuda.empty_cache()
            G.train()
        else:
            # Move data to the dest folder.
            path = [[path]]
            for _a in range(len(path)):
                fps_train_temp = [args.data_folder + '/' + task_ + '/Train' for task_ in os.listdir(args.data_folder)
                                  if int(os.path.split(task_)[-1].split('_')[1]) in path[_a]]
                for ell in range(len(path[_a])):
                    files = glob.glob(fps_train_temp[ell] + '/*')
                    file_idx = np.arange(len(files))
                    if args.subsample != 0:
                        permuted_idx = np.random.permutation(file_idx)
                        files_copy_idx = permuted_idx[:args.subsample]
                    else:
                        files_copy_idx = file_idx
                    for k in files_copy_idx:
                        shutil.copy2(files[k], dest)

    files = glob.glob(dest + '/*')
    new = []
    for k in range(len(files)):
        data = np.asarray(Image.open(files[k]), dtype="float32")
        data = data.reshape(1, 32, 32)
        data = (data - 127.5) / 127.5  # Normalize the images to [-1, 1]
        new.append(data)
        print(str(k))
    dataset_train = np.array(new)
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
