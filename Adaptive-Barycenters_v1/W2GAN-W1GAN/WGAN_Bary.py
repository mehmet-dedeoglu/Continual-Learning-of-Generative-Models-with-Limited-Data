import torch
import numpy as np
from torch.autograd import Variable
from torch import autograd
import time as t
import matplotlib.pyplot as plt
import os
plt.switch_backend('agg')


def get_infinite_batches(data_loader):
    while True:
        for i, (images) in enumerate(data_loader):
            yield images


class WGAN_GP(object):
    def __init__(self, args, save_folder, G, D, G_optim, D_optim):
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
        self.save_folder = save_folder
        self.workers = args.workers
        self.save_interval = args.save_interval
        self.multipliers = args.critic_weights
        print('WGAN object is successfully created.')

    def get_torch_variable(self, arg):
        if self.cuda:
            return Variable(arg).cuda(self.cuda_index)
        else:
            return Variable(arg)

    def check_cuda(self, cuda_flag=False):
        print('Cuda device: ' + str(cuda_flag))
        if cuda_flag == 0:
            self.cuda_index = 0
            self.cuda = True
            for n in range(len(self.D)):
                self.D[n].cuda(self.cuda_index)
            self.G.cuda(self.cuda_index)
            print("Cuda enabled flag: {}".format(self.cuda))
        else:
            self.cuda = False

    def train(self, train_loader):
        self.t_begin = t.time()

        iterator = []
        for i in range(len(train_loader)):
            iterator.append(get_infinite_batches(train_loader[i]))
        one = torch.tensor(1, dtype=torch.float)
        mone = one * -1
        if self.cuda:
            one = one.cuda(self.cuda_index)
            mone = mone.cuda(self.cuda_index)

        mult = self.multipliers
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
                gen_loss_save.append(self.G_train_step(len(iterator), mone, image_batch, True))
            gen_loss_save.append(self.G_train_step(len(iterator), mone, image_batch, True))
            if epo % self.save_interval == 0:
                # Save checkpoint
                checkpoint_folder = self.save_folder + '/checkpoints'
                if not os.path.exists(checkpoint_folder):
                    os.mkdir(checkpoint_folder)
                self.check_file = checkpoint_folder + '/checkpoint_Iter_' + str(epo)
                torch.save({
                    'modelG': self.G.state_dict(),
                    'modelD': [self.D[iii].state_dict() for iii in range(len(self.D))],
                    'optimizerG': self.g_optimizer.state_dict(),
                    'optimizerD': [self.d_optimizer[iii].state_dict() for iii in range(len(self.d_optimizer))]
                }, self.check_file)
                string_disc = checkpoint_folder + '/Negative_discriminator_loss'
                disc_loss_save_np = -np.array(disc_loss_save)
                np.savetxt(string_disc, disc_loss_save_np, delimiter=',')
                string_gen = checkpoint_folder + '/Negative_generator_loss'
                gen_loss_save_np = -np.array(gen_loss_save)
                np.savetxt(string_gen, gen_loss_save_np, delimiter=',')
        self.t_end = t.time()
        with open(checkpoint_folder + '/Time_Elapsed', 'w') as ff:
            ff.write(str(self.t_end - self.t_begin))
        return self.check_file

    def D_train_step(self, data_batch, d_num, back, print_bo):

        self.G.eval()
        for _k in range(d_num):
            self.D[_k].train()
            self.D[_k].zero_grad()

            data_real = self.get_torch_variable(data_batch[_k])
            d_loss_real = self.D[_k](data_real)
            d_loss_real = self.lamb[_k] * d_loss_real.mean()
            d_loss_real.backward(back[1])
            template = data_real.detach()

            noise = torch.randn((self.batch_size, self.noise_dim[0], self.noise_dim[1], self.noise_dim[2]))
            noise = self.get_torch_variable(noise)
            fake_images = self.G(noise, template)
            d_loss_fake = self.D[_k](fake_images)
            d_loss_fake = self.lamb[_k] * d_loss_fake.mean()
            d_loss_fake.backward(back[0])

            gradient_penalty = self.lamb[_k] * self.calculate_gradient_penalty(data_real.data, fake_images.data, _k)
            gradient_penalty.backward(retain_graph=True)

            # These two metrics are just for visualization.
            d_loss = d_loss_fake - d_loss_real + gradient_penalty
            Wasserstein_D = d_loss_real - d_loss_fake

            self.d_optimizer[_k].step()
            if print_bo:
                print(f'Disc. Index: {_k}, loss_fake: {d_loss_fake}, loss_real: {d_loss_real}, Wasserstein Distance: {Wasserstein_D}')
        self.G.train()

        return Wasserstein_D

    def G_train_step(self, d_num, mone, data_batch, print_bo):
        self.G.train()
        for k_k in range(d_num):
            self.D[k_k].eval()

        self.G.zero_grad()
        noise = torch.randn((self.batch_size, self.noise_dim[0], self.noise_dim[1], self.noise_dim[2]))
        noise = self.get_torch_variable(noise)
        fake_images = self.G(noise, self.get_torch_variable(data_batch[0]))
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
