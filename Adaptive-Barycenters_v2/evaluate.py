from Implementation.utils.config import parse_args_eval
import torch
import numpy as np
from PIL import Image
import os
from torch.autograd import Variable
from Implementation.archs.wb_networks import CNN_Gen_gp
import Implementation.supplementary.networks as networks


def evaluate(arg):

    checkpoint = torch.load(arg.checkpoint_path)

    seed = torch.randn((arg.sample_number, 100, 1, 1))
    # if arg.cuda == 'True':
    #     cuda_index = 0
    #     seed = Variable(seed).cuda(cuda_index)
    # else:
    #     seed = Variable(seed)

    if arg.select == 'W2GAN':
        g_mod = networks.get_g(args.config)
        g_mod.load_state_dict(checkpoint['modelG'])
        # template = np.asarray(Image.open(arg.template_path), dtype="float32")
        template = torch.randn((arg.sample_number, 1, 32, 32))
        cuda_index = 0
        seed = Variable(seed).cuda(cuda_index)
        data = g_mod(seed, template)
    else:
        g_mod = CNN_Gen_gp(arg.channel_number)
        g_mod.load_state_dict(checkpoint['modelG'])
        seed = Variable(seed)
        data = g_mod(seed)
    data = data.mul(0.5).add(0.5).data.cpu()

    data = data * 255
    data = np.asarray(data, dtype="uint8")
    if not os.path.exists(arg.data_folder):
        os.mkdir(arg.data_folder)
    for m in range(arg.sample_number):
        image_name = arg.data_folder + '/Image_' + str(m) + '_.png'
        im = Image.fromarray(data[m, 0, :, :])
        im.save(image_name)
        print('Printing image '+str(m))


if __name__ == '__main__':
    args = parse_args_eval()
    evaluate(args)
    print('Finished')