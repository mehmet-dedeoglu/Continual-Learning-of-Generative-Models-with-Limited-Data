import torch
import numpy as np
from torch.autograd import Variable
from torch import autograd
import time as t
import matplotlib.pyplot as plt
import os
import supplementary.utils
import supplementary.losses
import functools
plt.switch_backend('agg')


def get_infinite_batches(data_loader):
    while True:
        for i, (images) in enumerate(data_loader):
            yield images


class W2GAN_GP(object):
    def __init__(self, args, save_folder, G, D_forward, D_backward, G_optim, D_optim):
        self.G = G
        self.D_forward = D_forward
        self.D_backward = D_backward
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
        self.config = args.config
        self.cost = functools.partial(losses.cost, l=args.config.l, p=args.config.p)
        print('W2GAN object is successfully created.')

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
            for n in range(len(self.D_forward)):
                self.D_forward[n].cuda(self.cuda_index)
                self.D_backward[n].cuda(self.cuda_index)
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

        disc_forw_loss_save = []
        disc_back_loss_save = []
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
            for ii in range(len(self.D_forward)):
                for p in self.D_forward[ii].parameters():
                    p.requires_grad = True
                for pp in self.D_backward[ii].parameters():
                    pp.requires_grad = True
            # Update Discriminators
            for _ in range(self.critic_iter-1):
                disc_losses = self.D_train_step(image_batch, len(iterator), [one, mone], True)
                disc_back_loss_save.append(disc_losses)
                disc_forw_loss_save.append(disc_losses)
            disc_losses = self.D_train_step(image_batch, len(iterator), [one, mone], True)
            disc_back_loss_save.append(disc_losses)
            disc_forw_loss_save.append(disc_losses)

            for ii in range(len(self.D_forward)):
                for p in self.D_forward[ii].parameters():
                    p.requires_grad = False
                for pp in self.D_backward[ii].parameters():
                    pp.requires_grad = False
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
                    'modelDF': [self.D_forward[iii].state_dict() for iii in range(len(self.D_forward))],
                    'modelDB': [self.D_backward[iii].state_dict() for iii in range(len(self.D_backward))],
                    'optimizerG': self.g_optimizer.state_dict(),
                    'optimizerD': [self.d_optimizer[iii].state_dict() for iii in range(len(self.d_optimizer))]
                }, self.check_file)
                string_disc = checkpoint_folder + '/Negative_discriminator_loss'
                disc_loss_save_np = -np.array(disc_forw_loss_save)
                np.savetxt(string_disc, disc_loss_save_np, delimiter=',')
                string_gen = checkpoint_folder + '/Negative_generator_loss'
                gen_loss_save_np = -np.array(gen_loss_save)
                np.savetxt(string_gen, gen_loss_save_np, delimiter=',')
        self.t_end = t.time()
        with open(checkpoint_folder + '/Time_Elapsed', 'w') as ff:
            ff.write(str(self.t_end - self.t_begin))
        return self.check_file

    def psi(self, y, _k):
        return -self.D_forward[_k](y) + self.D_backward[_k](y)

    def get_tx(self, x, _k, reverse=False):
        x = Variable(x.data, requires_grad=True)
        if reverse:
            ux = self.psi(x, _k)
        else:
            ux = self.D_forward[_k](x)
        dux = torch.autograd.grad(outputs=ux, inputs=x,
                                  grad_outputs=utils.get_ones(ux.size()),
                                  create_graph=True, retain_graph=True,
                                  only_inputs=True)[0]
        Tx = x - dux
        return Tx

    def calc_dloss(self, x, y, tx, ty, ux, vy, _k):
        d_loss = -torch.mean(ux + vy)
        if self.config.ineq:
            d_loss += losses.ineq_loss(x, y, ux, vy, self.cost, self.config.lambda_ineq)
        if self.config.ineq_interp:
            d_loss += losses.calc_interp_ineq(x, y, self.D_forward[_k], self.psi, _k, self.cost,
                                              self.config.lambda_ineq, losses.ineq_loss)
        if self.config.eq_phi:
            d_loss += losses.calc_eq(x, tx, self.D_forward[_k], self.psi, _k,
                                     self.cost, self.config.lambda_eq)
        if self.config.eq_psi:
            d_loss += losses.calc_eq(ty, y, self.D_forward[_k], self.psi, _k,
                                     self.cost, self.config.lambda_eq)
        return self.lamb[_k] * d_loss

    def D_train_step(self, data_batch, d_num, back, print_bo):

        self.G.eval()
        for _k in range(d_num):
            self.D_forward[_k].train()
            self.D_backward[_k].train()
            self.d_optimizer[_k].zero_grad()

            data_real = self.get_torch_variable(data_batch[_k])
            noise = torch.randn((self.batch_size, self.noise_dim[0], self.noise_dim[1], self.noise_dim[2]))
            noise = self.get_torch_variable(noise)
            fake_images = self.G(noise, data_real)

            x, y = data_real.detach(), fake_images.detach()
            tx, ty = self.get_tx(x, _k), self.get_tx(y, _k, reverse=True)
            ux, vy = self.D_forward[_k](x), self.psi(y, _k)
            d_loss = self.calc_dloss(x, y, tx, ty, ux, vy, _k)
            d_loss.backward()
            self.d_optimizer[_k].step()
            self.d_loss = d_loss.data.item()

            if print_bo:
                print(f'Disc. Index: {_k}, Loss: {self.d_loss}')
        self.G.train()

        return self.d_loss.clone().detach().cpu().numpy()

    def calc_gloss(self, x, y, ux, vy, config):
        return torch.mean(vy)

    def G_train_step(self, d_num, mone, data_batch, print_bo):
        self.G.train()
        for k_k in range(d_num):
            self.D_forward[k_k].eval()
            self.D_backward[k_k].eval()
        self.g_optimizer.zero_grad()

        noise = torch.randn((self.batch_size, self.noise_dim[0], self.noise_dim[1], self.noise_dim[2]))
        noise = self.get_torch_variable(noise)
        fake_images = self.G(noise, self.get_torch_variable(data_batch[0]))

        for kk in range(d_num-1):
            data_real = self.get_torch_variable(data_batch[kk])
            x, y = data_real, fake_images
            ux, vy = self.D_forward[kk](x), self.psi(y, kk)
            g_loss = self.lamb[kk] * self.calc_gloss(x, y, ux, vy, self.config)
            g_loss.backward(retain_graph=True)
        data_real = self.get_torch_variable(data_batch[d_num-1])
        x, y = data_real, fake_images
        ux, vy = self.D_forward[d_num-1](x), self.psi(y, d_num-1)
        g_loss = self.lamb[d_num-1] * self.calc_gloss(x, y, ux, vy, self.config)
        g_loss.backward()
        self.g_optimizer.step()
        self.g_loss = g_loss.data.item()

        print(f'g_COST: {self.g_loss}')
        for k_k in range(d_num):
            self.D_forward[k_k].train()
            self.D_backward[k_k].train()

        return g_loss.clone().detach().cpu().numpy()

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
