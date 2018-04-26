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

from model.wae import *

from utils import *
#torch.load('my_file.pt', map_location=lambda storage, loc: storage)

DIM = 64 # Model dimensionality
BATCH_SIZE = 50 # Batch size
CRITIC_ITERS = 5 # For WGAN and WGAN-GP, number of critic iters per gen iter
LAMBDA = 10 # Gradient penalty lambda hyperparameter
ITERS = 200000 # How many generator iterations to train for
OUTPUT_DIM = 1080 # Number of pixels in MNIST (28*28)
EPSILON = 1e-15


torch.manual_seed(1)
use_cuda = torch.cuda.is_available()
if use_cuda:
    gpu = 0


data_test = np.load('./data/train_data_nolabel.npy')
data_loader = torch.utils.data.DataLoader(data_test,
                                          batch_size=BATCH_SIZE,
                                          shuffle=True)


Q = Encoder()
P = Decoder()
D = Discriminator()

# Optimizers
P_optim = optim.Adam(P.parameters(), lr = 0.001)
Q_enc_optim = optim.Adam(Q.parameters(), lr = 0.001)
Q_gen_optim = optim.Adam(Q.parameters(), lr = 0.001)
D_optim = optim.Adam(D.parameters(), lr = 0.001)


D_cost_list = []
G_cost_list = []
W_D_list = []

for iteration in range(ITERS):
    print('Iteration:',iteration)
    step = 0
    for i, data_i in enumerate(data_loader):
        start_time = time.time()

        P.zero_grad()
        Q.zero_grad()
        D.zero_grad()

        real_data_v = Variable(data_i).float()

        if use_cuda:
            real_data_v = real_data_v.cuda()

        batch_size = real_data_v.size()[0]
        real_data_v = real_data_v.view(batch_size, -1)

        z_sample = Q.forward(real_data_v)
        x_sample = P.forward(z_sample)
        recon_loss = F.binary_cross_entropy(x_sample + EPSILON, real_data_v + EPSILON)
        recon_loss.backward()

        P_optim.step()
        Q_enc_optim.step()

        Q.eval()
        z_real_gauss = Variable(torch.randn(real_data_v.size()[0], Z_DIM) * 5.)
        D_real_gauss = D.forward(z_real_gauss)

        z_fake_gauss = Q.forward(real_data_v)
        D_fake_gauss = D.forward(z_fake_gauss)

        D_loss = -LAMBDA * torch.mean(torch.log(D_real_gauss + EPSILON) + torch.log(1-D_fake_gauss+EPSILON))
        D_loss.backward()
        D_optim.step()

        Q.train()
        z_fake_gauss = Q.forward(real_data_v)
        D_fake_gauss = D.forward(z_fake_gauss)

        G_loss = -LAMBDA*torch.mean(torch.log(D_fake_gauss + EPSILON))
        G_loss.backward()
        Q_gen_optim.step()

        step += 1

        if i % 100 == 0:
            print("Epoch: %d, Step: [%d/%d], Reconstruction Loss: %.4f, Discriminator Loss: %.4f, Generator Loss: %.4f" %
                  (iteration + 1, step + 1, len(data_loader), recon_loss.data[0], D_loss.data[0], G_loss.data[0]))
            noise = torch.randn(1, 128)
            noisev = autograd.Variable(noise)
            recover_sound(P.forward(noisev),iteration,i)

        #if i % 100 == 0:
        #    D_cost_i = D_cost.cpu().data.numpy()
        #    G_cost_i = G_cost.data.cpu().numpy()
        #    W_D = Wasserstein_D.cpu().data.numpy()
        #    D_cost_list.append(D_cost_i)
        #    G_cost_list.append(G_cost_i)
        #    W_D_list.append(W_D)
        #    print('D_cost:',D_cost_i)
        #    print('G_cost:',G_cost_i)
        #    print('Wasserstein distance:', W_D)
        #    np.save('D_cost', D_cost_list)
        #    np.save('G_cost', G_cost_list)
        #    np.save('W_D', W_D_list)
        #    noise = torch.randn(1,128)
        ##    noisev = autograd.Variable(noise)
         #   recover_sound(netG(noisev),iteration,i)
        #if Wasserstein_D.data.numpy() < BEST_WD:
         #   BEST_WD = Wasserstein_D.data.numpy()
          #  torch.save(netD, 'D_model')
           # torch.save(netG, 'G_model')


