import torch
import torch.utils.data as data_utils
import numpy as np
from torch.autograd import Variable
from torch import autograd
import time as t
import matplotlib.pyplot as plt
import os
from torchvision import utils
from PIL import Image
import glob
plt.switch_backend('agg')


def get_infinite_batches(data_loader):
    while True:
        for i, (images) in enumerate(data_loader):
            yield images


class WGAN_GP(object):
    def __init__(self, args, save_folder, G, D, G_optim, D_optim, fid_state='Empty'):
        self.G = G
        self.D = D
        self.check_cuda(args.cuda)
        self.batch_size = args.batch_size
        self.d_optimizer = D_optim
        self.g_optimizer = G_optim
        self.generator_iters = args.generator_iters
        self.critic_iter = args.critic_iter
        self.lambda_term = args.lambda_val
        self.EPOCHS = args.epoch_num
        self.noise_dim = args.noise_dim
        self.FID = fid_state
        self.FID_folder = save_folder + '/FID_Images'
        self.save_folder = save_folder
        self.FID_BATCH_SIZE = args.FID_BATCH_SIZE
        self.workers = args.workers
        self.save_interval = args.save_interval

    def get_torch_variable(self, arg):
        if self.cuda:
            return Variable(arg).cuda(self.cuda_index)
        else:
            return Variable(arg)

    def check_cuda(self, cuda_flag=False):
        print(cuda_flag)
        if cuda_flag:
            self.cuda_index = 0
            self.cuda = True
            for n in range(len(self.D)):
                self.D[n].cuda(self.cuda_index)
            self.G.cuda(self.cuda_index)
            print("Cuda enabled flag: {}".format(self.cuda))
        else:
            self.cuda = False

    def train(self, train_loader, fid_data, discriminator_save_iter=-1):
        self.t_begin = t.time()
        if self.FID == 'FID':
            fid_iter = get_infinite_batches(fid_data[0])

        iterator = []
        for i in range(len(train_loader)):
            iterator.append(get_infinite_batches(train_loader[i]))
        one = torch.tensor(1, dtype=torch.float)
        mone = one * -1
        if self.cuda:
            one = one.cuda(self.cuda_index)
            mone = mone.cuda(self.cuda_index)

        mult = [1, 15, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        lamb_sum = 0
        self.lamb = []
        for dn in range(len(iterator)):
            lamb_sum += mult[dn]
        for dm in range(len(iterator)):
            self.lamb.append(len(iterator)*mult[dm]/lamb_sum)

        disc_loss_save = []
        gen_loss_save = []
        for epo in range(self.EPOCHS):
            image_batch = []
            for j in range(len(iterator)):
                image_batch.append(iterator[j].__next__())
                # Check for batch to have full batch_size
                if image_batch[j].size()[0] != self.batch_size:
                    # continue
                    image_batch[j] = iterator[j].__next__()
            print('Epoch: ' + str(epo))
            # Requires grad, Generator requires_grad = False
            for ii in range(len(self.D)):
                for p in self.D[ii].parameters():
                    p.requires_grad = True
            # Update Discriminators
            for _ in range(self.critic_iter-1):
                disc_loss_save.append(self.D_train_step(image_batch, len(iterator), [one, mone], True))
            disc_loss_save.append(self.D_train_step(image_batch, len(iterator), [one, mone], True))

            for ii in range(len(self.D)):
                for p in self.D[ii].parameters():
                    p.requires_grad = False
            for _ in range(self.generator_iters-1):
                gen_loss_save.append(self.G_train_step(len(iterator), mone, True))
            gen_loss_save.append(self.G_train_step(len(iterator), mone, True))
            if epo % self.save_interval == 0:
                # Save checkpoint
                self.check_file = self.save_folder + '/checkpoint_Iter_'+str(epo)
                torch.save({
                    'modelG': self.G,
                    'modelD': self.D,
                    'optimizerG': self.g_optimizer,
                    'optimizerD': self.d_optimizer
                }, self.check_file)
                self.fid_iter = self.FID_folder + '/Iteration_' + str(epo)
                self.fid_fake = self.fid_iter + '/Fake_Images'
                self.fid_real = self.fid_iter + '/Real_Images'
                if not os.path.exists(self.FID_folder):
                    os.mkdir(self.FID_folder)
                if not os.path.exists(self.fid_iter):
                    os.mkdir(self.fid_iter)
                    os.mkdir(self.fid_fake)
                    os.mkdir(self.fid_real)
                _ = self.generate(inception=True, samp_num=self.FID_BATCH_SIZE, iter=epo)
                if self.FID == 'FID':
                    images_real = fid_iter.__next__().numpy()
                    if images_real.shape[0] != self.FID_BATCH_SIZE:
                        images_real = fid_iter.__next__().numpy()
                    generated_image = (images_real + 1) * 255/2
                    data = np.asarray(generated_image, dtype="uint8")
                    data = np.transpose(data, [0, 2, 3, 1])
                    for m in range(self.FID_BATCH_SIZE):
                        image_name = self.fid_real + '/Image_' + str(m) + '_.png'
                        im = Image.fromarray(data[m, :, :, :])
                        im.save(image_name)
                string_disc = self.save_folder + 'Negative_discriminator_loss'
                disc_loss_save_np = -np.array(disc_loss_save)
                np.savetxt(string_disc, disc_loss_save_np, delimiter=',')
                string_gen = self.save_folder + 'Negative_generator_loss'
                gen_loss_save_np = -np.array(gen_loss_save)
                np.savetxt(string_gen, gen_loss_save_np, delimiter=',')
        self.t_end = t.time()
        with open(self.save_folder + 'Time_Elapsed', 'w') as ff:
            ff.write(str(self.t_end - self.t_begin))
        return self.check_file

    def generate(self, gen=None, samp_num=1000, inception=False, iter=0, fld_str=''):
        seed = torch.randn((samp_num, self.noise_dim[0], self.noise_dim[1], self.noise_dim[2]))
        seed = self.get_torch_variable(seed)
        if gen is None:
            self.G.eval()
            generated_image = self.G(seed)
            generated_image = generated_image.mul(0.5).add(0.5).data.cpu()

            generated_image = generated_image * 255
            data = np.asarray(generated_image, dtype="uint8")
            data = np.transpose(data, [0, 2, 3, 1])
            for m in range(samp_num):
                image_name = self.fid_fake + '/Image_' + str(m) + '_.png'
                im = Image.fromarray(data[m, :, :, :])
                im.save(image_name)
            self.G.train()
        else:
            gen.eval()
            mm = 0
            for split in range(100):
                generated_image = gen(seed[int(split*samp_num/100):int((split+1)*samp_num/100)])
                generated_image = generated_image.mul(0.5).add(0.5).data.cpu()
                generated_image = generated_image * 255
                data = np.asarray(generated_image, dtype="uint8")
                data = np.transpose(data, [0, 2, 3, 1])
                gen_folder = fld_str
                if not os.path.exists(gen_folder):
                    os.mkdir(gen_folder)
                for m in range(int(samp_num/100)):
                    image_name = gen_folder + '/Image_' + str(mm) + '_.png'
                    im = Image.fromarray(data[m, :, :, :])
                    im.save(image_name)
                    mm += 1
                del generated_image, data, im
                torch.cuda.empty_cache()
            #
            files = glob.glob(gen_folder + '/*')
            new = []
            for k in range(len(files)):
                data = np.asarray(Image.open(files[k]), dtype="float32")
                data = np.transpose(data, [2, 0, 1])
                data = (data - 127.5) / 127.5  # Normalize the images to [-1, 1]
                new.append(data)
                print(str(k))
            dataset_train = np.array(new)
            data = torch.utils.data.DataLoader(dataset_train, batch_size=self.batch_size, shuffle=True,
                                               num_workers=self.workers, pin_memory=True)
            gen.train()
        return data

    def D_train_step(self, data_batch, d_num, back, print_bo):

        self.G.eval()
        for _k in range(d_num):
            self.D[_k].train()
            self.D[_k].zero_grad()

            data_real = self.get_torch_variable(data_batch[_k])
            d_loss_real = self.D[_k](data_real)
            d_loss_real = self.lamb[_k] * d_loss_real.mean()
            d_loss_real.backward(back[1])

            noise = torch.randn((self.batch_size, self.noise_dim[0], self.noise_dim[1], self.noise_dim[2]))
            noise = self.get_torch_variable(noise)
            fake_images = self.G(noise)
            d_loss_fake = self.D[_k](fake_images)
            d_loss_fake = self.lamb[_k] * d_loss_fake.mean()
            d_loss_fake.backward(back[0])

            gradient_penalty = self.lamb[_k] * self.calculate_gradient_penalty(data_real.data, fake_images.data, _k)
            gradient_penalty.backward(retain_graph=True)

            # These two metrics are just for visualization.
            d_loss = d_loss_fake - d_loss_real + gradient_penalty
            # d_loss.backward(back[0], retain_graph=True)
            Wasserstein_D = d_loss_real - d_loss_fake

            self.d_optimizer[_k].step()
            if print_bo:
                print(f'Disc. Index: {_k}, loss_fake: {d_loss_fake}, loss_real: {d_loss_real}, Wasserstein Distance: {Wasserstein_D}')
        self.G.train()

        return Wasserstein_D

    def G_train_step(self, d_num, mone, print_bo):
        self.G.train()
        for k_k in range(d_num):
            self.D[k_k].eval()

        self.G.zero_grad()
        noise = torch.randn((self.batch_size, self.noise_dim[0], self.noise_dim[1], self.noise_dim[2]))
        noise = self.get_torch_variable(noise)
        fake_images = self.G(noise)
        for kk in range(d_num-1):
            g_loss = self.D[kk](fake_images)
            g_loss = self.lamb[kk] * g_loss.mean()
            g_loss.backward(mone, retain_graph=True)
        g_loss = self.D[d_num-1](fake_images)
        g_loss = self.lamb[d_num-1] * g_loss.mean()
        g_loss.backward(mone)
        g_cost = -g_loss
        self.g_optimizer.step()
        print(f'g_COST: {g_cost}')
        for k_k in range(d_num):
            self.D[k_k].train()

        return g_loss

    def calculate_gradient_penalty(self, real_images, fake_images, iter):
        eta = torch.FloatTensor(self.batch_size, 1, 1, 1).uniform_(0, 1)
        eta = eta.expand(self.batch_size, real_images.size(1), real_images.size(2), real_images.size(3))
        if self.cuda:
            eta = eta.cuda(self.cuda_index)
        else:
            eta = eta
        interpolated = eta * real_images + ((1 - eta) * fake_images)
        if self.cuda:
            interpolated = interpolated.cuda(self.cuda_index)
        else:
            interpolated = interpolated
        # define it to calculate gradient
        interpolated = Variable(interpolated, requires_grad=True)
        # calculate probability of interpolated examples
        prob_interpolated = self.D[iter](interpolated)
        # calculate gradients of probabilities with respect to examples
        gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                                  grad_outputs=torch.ones(
                                      prob_interpolated.size()).cuda(self.cuda_index) if self.cuda else torch.ones(
                                      prob_interpolated.size()),
                                  create_graph=True, retain_graph=True)[0]
        grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.lambda_term
        return grad_penalty
