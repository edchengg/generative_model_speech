import time

import matplotlib
matplotlib.use('Agg')
import numpy as np

import torch
import torch.utils.data
from torch.autograd import Variable
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from model.wgan import *
from utils import *
#torch.load('my_file.pt', map_location=lambda storage, loc: storage)

DIM = 64 # Model dimensionality
BATCH_SIZE = 50 # Batch size
CRITIC_ITERS = 5 # For WGAN and WGAN-GP, number of critic iters per gen iter
LAMBDA = 10 # Gradient penalty lambda hyperparameter
ITERS = 200000 # How many generator iterations to train for
OUTPUT_DIM = 1080 # Number of pixels in MNIST (28*28)

torch.manual_seed(1)
use_cuda = torch.cuda.is_available()
if use_cuda:
    gpu = 0


data_test = np.load('train_data_nolabel.npy')
data_loader = torch.utils.data.DataLoader(data_test,
                                          batch_size=BATCH_SIZE,
                                          shuffle=True)

netG = Generator()
netD = Discriminator()




if use_cuda:
    netD = netD.cuda()
    netG = netG.cuda()

optimizerD = optim.Adam(netD.parameters(), lr=1e-4, betas=(0, 0.9))
optimizerG = optim.Adam(netG.parameters(), lr=1e-4, betas=(0, 0.9))


def calc_gradient_penalty(netD, real_data, fake_data):

    # uniform distribution
    alpha = torch.rand(real_data.size())

    alpha = alpha.cuda(gpu) if use_cuda else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if use_cuda:
        interpolates = interpolates.cuda()
    interpolates = Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(gpu) if use_cuda else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty


D_cost_list = []
G_cost_list = []
W_D_list = []

BEST_WD = 999999999
for iteration in range(ITERS):
    print('Iteration:',iteration)
    for i, data_i in enumerate(data_loader):
        start_time = time.time()
        ############################
        # (1) Update D network
        ###########################
        for p in netD.parameters():  # reset requires_grad
            p.requires_grad = True  # they are set to False below in netG update

        real_data_v = Variable(data_i).float()

        if use_cuda:
            real_data_v = real_data_v.cuda()

        netD.zero_grad()

        # train with real

        D_real = netD(real_data_v)
        D_real = D_real.mean()
        # print D_real
        D_real.backward()
        #print(real_data_v.size()[0])

        # train with fake
        noise = torch.randn(real_data_v.size()[0], 128)
        if use_cuda:
            noise = noise.cuda(gpu)
        noisev = autograd.Variable(noise, volatile=True)  # totally freeze netG
        fake = autograd.Variable(netG(noisev).data)
        inputv = fake
        D_fake = netD(inputv)
        D_fake = D_fake.mean()
        D_fake.backward()

        # train with gradient penalty
        gradient_penalty = calc_gradient_penalty(netD, real_data_v.data, fake.data)
        gradient_penalty.backward()

        D_cost = D_fake - D_real + gradient_penalty
        Wasserstein_D = D_real - D_fake
        optimizerD.step()


        ############################
        # (2) Update G network
        ###########################
        if i % CRITIC_ITERS == 0:
            for p in netD.parameters():
                p.requires_grad = False  # to avoid computation
            netG.zero_grad()

            noise = torch.randn(BATCH_SIZE, 128)
            if use_cuda:
                noise = noise.cuda(gpu)
            noisev = autograd.Variable(noise)
            fake = netG(noisev)
            G = netD(fake)
            G = G.mean()
            G.backward()
            G_cost = -G
            optimizerG.step()

        if i % 100 == 0:
            D_cost_i = D_cost.cpu().data.numpy()
            G_cost_i = G_cost.data.cpu().numpy()
            W_D = Wasserstein_D.cpu().data.numpy()
            D_cost_list.append(D_cost_i)
            G_cost_list.append(G_cost_i)
            W_D_list.append(W_D)
            print('D_cost:',D_cost_i)
            print('G_cost:',G_cost_i)
            print('Wasserstein distance:', W_D)
            np.save('D_cost', D_cost_list)
            np.save('G_cost', G_cost_list)
            np.save('W_D', W_D_list)
            noise = torch.randn(1,128)
            noisev = autograd.Variable(noise)
            recover_sound(netG(noisev),iteration,i)
        if Wasserstein_D.data.numpy() < BEST_WD:
            BEST_WD = Wasserstein_D.data.numpy()
            torch.save(netD, 'D_model')
            torch.save(netG, 'G_model')
        # Write logs and save samples
        #lib.plot.plot('tmp/speech/time', time.time() - start_time)
        #lib.plot.plot('tmp/speech/train disc cost', D_cost.cpu().data.numpy())
        #lib.plot.plot('tmp/speech/train gen cost', G_cost.cpu().data.numpy())
        #lib.plot.plot('tmp/speech/wasserstein distance', Wasserstein_D.cpu().data.numpy())




