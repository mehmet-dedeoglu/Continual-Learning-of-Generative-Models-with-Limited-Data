import os
import torch
from config import parse_args
import networks
from torch.autograd import Variable
from DNN_MNIST import CNN_Gen_gp
import numpy as np
from PIL import Image
import shutil
import glob


def move_data(args, cls, samp_num, folder_name):
    class_ = args.train_classes[cls]
    if type(class_) is not str:
        # Move data to the destination folder.
        fps_train_temp = [args.data_folder + '/' + task_ + '/Train' for task_ in os.listdir(args.data_folder)
                          if int(os.path.split(task_)[-1].split('_')[1]) in [class_]]
        for ell in range(len(fps_train_temp)):
            files = glob.glob(fps_train_temp[ell] + '/*')
            file_idx = np.arange(len(files))
            permuted_idx = np.random.permutation(file_idx)
            files_copy_idx = permuted_idx[:min(len(files), samp_num)]
            for kk in files_copy_idx:
                shutil.copy2(files[kk], folder_name)
    else:
        print('Error!!!')


args = parse_args()
input_folder = 'MNIST_Simulations/stage27/checkpoints'
# class_line = 13
classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# classes = [8, 9]

if not os.path.exists(input_folder.split('/')[0] + '/FID_path'):
    os.mkdir(input_folder.split('/')[0] + '/FID_path')
destination_path = input_folder.split('/')[0] + '/FID_path/' + input_folder.split('/')[1]
if not os.path.exists(destination_path):
    os.mkdir(destination_path)
chk_files = os.listdir(input_folder)
samp_num = 1000
print('First, fake images will be generated...')
for k in range(len(chk_files)):
    checkpoint = torch.load(input_folder + '/' + chk_files[k])
    seed = torch.randn((samp_num, args.noise_dim[0], args.noise_dim[1], args.noise_dim[2]))
    if args.select == 'W2GAN':
        g_mod = networks.get_g(args.config)
        g_mod.load_state_dict(checkpoint['modelG'])
        # template = np.asarray(Image.open(arg.template_path), dtype="float32")
        template = torch.randn((samp_num, 1, 32, 32))
        cuda_index = 0
        seed = Variable(seed).cuda(cuda_index)
        data = g_mod(seed, template)
    else:
        g_mod = CNN_Gen_gp(args.channel_number)
        g_mod.load_state_dict(checkpoint['modelG'])
        seed = Variable(seed)
        data = g_mod(seed)
    data = data.mul(0.5).add(0.5).data.cpu()

    data = data * 255
    data = np.asarray(data, dtype="uint8")
    fake_folder = destination_path + '/fake_images'
    if not os.path.exists(fake_folder):
        os.mkdir(fake_folder)
    iter_folder = fake_folder + '/Iteration_' + str(chk_files[k].split('_')[-1])
    if not os.path.exists(iter_folder):
        os.mkdir(iter_folder)
    for m in range(samp_num):
        image_name = iter_folder + '/Image_' + str(m) + '_.png'
        im = Image.fromarray(data[m, 0, :, :])
        im.save(image_name)
    print('Completed printing for iteration '+str(chk_files[k].split('_')[-1]))

print('Fake images are generated. Now, real images will be generated...')
real_folder = destination_path + '/real_images'
if not os.path.exists(real_folder):
    os.mkdir(real_folder)
else:
    shutil.rmtree(real_folder)
    os.mkdir(real_folder)

for c in range(len(classes)):
    move_data(args, classes[c], int(samp_num/len(classes)), real_folder)
print('Image generation is completed.')
