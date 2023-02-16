from config import parse_args
import torch
import os
import numpy as np
from torch.autograd import Variable
import glob
from PIL import Image


def get_torch_variable(arg, val):
    if arg.cuda:
        return Variable(val).cuda(0)
    else:
        return Variable(val)


if __name__ == '__main__':
    # Get args
    args = parse_args()
    # Select a set of seed values
    seed_size = 1000
    seed = torch.randn((seed_size, 100, 1, 1))
    seed = get_torch_variable(args, seed)
    # Generate images
    model_loc = 'MNIST_Simulations_NonOverlapping/Folder_[2 9]_Ensemble'
    model_fps = [model_loc + '/' + task_ for task_ in os.listdir(model_loc)]
    # Create image folders
    image_fld = model_loc + '/Images'
    if not os.path.exists(image_fld):
        os.mkdir(image_fld)
    for i in range(len(model_fps)):
        iter_num = int(os.path.split(model_fps[i])[-1].split('_')[-1])
        print('Iteration: ' + str(iter_num))
        # Import generator model
        checkpoint = torch.load(model_fps[i])
        G = checkpoint['modelG']
        G.eval()
        # Generate images using the pre-determined seed
        generated_image = G(seed)
        generated_image = generated_image.mul(0.5).add(0.5).data.cpu()
        generated_image = generated_image * 255
        data = np.asarray(generated_image, dtype="uint8")
        # Save images under designated folder
        iter_fld = image_fld + '/Iteration_' + str(iter_num)
        if not os.path.exists(iter_fld):
            os.mkdir(iter_fld)
        for j in range(seed_size):
            image_name = iter_fld + '/Image_' + str(j) + '_.png'
            im = Image.fromarray(data[j, 0, :, :])
            im.save(image_name)
