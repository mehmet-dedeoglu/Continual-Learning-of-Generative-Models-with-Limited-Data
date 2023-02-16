from config import parse_args_eval
import torch
import numpy as np
from PIL import Image
import os
from torch.autograd import Variable


def evaluate(arg):

    checkpoint = torch.load(arg.checkpoint_path)
    model_G = checkpoint['modelG']

    seed = torch.randn((arg.sample_number, 100, 1, 1))
    if arg.cuda == 'True':
        cuda_index = 0
        seed = Variable(seed).cuda(cuda_index)
    else:
        seed = Variable(seed)

    data = model_G(seed)
    data = data.mul(0.5).add(0.5).data.cpu()

    data = data * 255
    data = np.asarray(data, dtype="uint8")
    data = np.transpose(data, [0, 2, 3, 1])
    if not os.path.exists(arg.data_folder):
        os.mkdir(arg.data_folder)
    for m in range(arg.sample_number):
        image_name = arg.data_folder + '/Image_' + str(m) + '_.png'
        im = Image.fromarray(data[m, :, :, :])
        im.save(image_name)
        print('Printing image '+str(m))


if __name__ == '__main__':
    args = parse_args_eval()
    evaluate(args)
    print('Finished')