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

Z_DIM = 64 # Model dimensionality
BATCH_SIZE = 128 # Batch size
CRITIC_ITERS = 5 # For WGAN and WGAN-GP, number of critic iters per gen iter
LAMBDA = 10 # Gradient penalty lambda hyperparameter
ITERS = 20 # How many generator iterations to train for
OUTPUT_DIM = 760 # Number of pixels in MNIST (28*28)
EPSILON = 1e-15


torch.manual_seed(1)
use_cuda = torch.cuda.is_available()
if use_cuda:
    gpu = 0


data_test = np.load('./data/traindata_19_40_nolabel.npy')
data_loader = torch.utils.data.DataLoader(data_test,
                                          batch_size=BATCH_SIZE,
                                          shuffle=True)

data_valid = np.load('./data/devdata_19_40_nolabel.npy')
valid_loader = torch.utils.data.DataLoader(data_test,
                                          batch_size=BATCH_SIZE,
                                          shuffle=True)

data_fix = np.load('./data/traindata_19_40_nolabel.npy')
fix_loader = torch.utils.data.DataLoader(data_fix,
                                          batch_size=1,
                                          shuffle=False)

fix_iter = iter(fix_loader)
_= next(fix_iter)
fixed_x = next(fix_iter)
fixed_x = Variable(fixed_x).float()

if use_cuda:
    fixed_x = fixed_x.cuda()
recover_sound(fixed_x,'real','wae')


encoder = Encoder(Z_DIM)
decoder = Decoder(Z_DIM)
discriminator = Discriminator(Z_DIM)

# Optimizers
P_optim = optim.Adam(decoder.parameters(), lr = 0.001)
Q_enc_optim = optim.Adam(encoder.parameters(), lr = 0.001)
Q_gen_optim = optim.Adam(encoder.parameters(), lr = 0.001)
D_optim = optim.Adam(discriminator.parameters(), lr = 0.001)


def evaluate(valid_data):
    valid_recon = 0
    valid_d = 0
    valid_g = 0
    for i, data_i in enumerate(valid_data):
        real_data_v = Variable(data_i).float()

        if use_cuda:
            real_data_v = real_data_v.cuda()

        batch_size = real_data_v.size()[0]
        real_data_v = real_data_v.view(batch_size, -1)

        z_sample = encoder.forward(real_data_v)
        x_sample = decoder.forward(z_sample)
        recon_loss = F.binary_cross_entropy(x_sample + EPSILON, F.sigmoid(real_data_v) + EPSILON)


        z_real_gauss = Variable(torch.randn(real_data_v.size()[0], Z_DIM) * 5.)
        D_real_gauss = discriminator.forward(z_real_gauss)

        z_fake_gauss = encoder.forward(real_data_v)
        D_fake_gauss = discriminator.forward(z_fake_gauss)

        D_loss = -LAMBDA * torch.mean(torch.log(D_real_gauss + EPSILON) + torch.log(1 - D_fake_gauss + EPSILON))

        z_fake_gauss = encoder.forward(real_data_v)
        D_fake_gauss = discriminator.forward(z_fake_gauss)

        G_loss = -LAMBDA * torch.mean(torch.log(D_fake_gauss + EPSILON))

        valid_recon += recon_loss
        valid_d += D_loss
        valid_g += G_loss
    return [valid_recon/len(valid_data),valid_d/len(valid_data)/BATCH_SIZE,valid_g/len(valid_data)/BATCH_SIZE]


def inverse_sigmoid(input):
    res = -1 * torch.log(1 / input - 1)
    return res


for iteration in range(ITERS):
    step = 0
    train_loss_recon = 0
    train_D_loss = 0
    train_G_loss = 0
    encoder.train()
    decoder.train()
    discriminator.train()
    for i, data_i in enumerate(data_loader):
        start_time = time.time()

        decoder.zero_grad()
        encoder.zero_grad()
        discriminator.zero_grad()

        real_data_v = Variable(data_i).float()

        if use_cuda:
            real_data_v = real_data_v.cuda()

        batch_size = real_data_v.size()[0]
        real_data_v = real_data_v.view(batch_size, -1)

        z_sample = encoder.forward(real_data_v)
        x_sample = decoder.forward(z_sample)
        recon_loss = F.binary_cross_entropy(x_sample + EPSILON, F.sigmoid(real_data_v) + EPSILON)
        recon_loss.backward()

        P_optim.step()
        Q_enc_optim.step()

        encoder.eval()
        z_real_gauss = Variable(torch.randn(real_data_v.size()[0], Z_DIM) * 5.)
        D_real_gauss = discriminator.forward(z_real_gauss)

        z_fake_gauss = encoder.forward(real_data_v)
        D_fake_gauss = discriminator.forward(z_fake_gauss)

        D_loss = -LAMBDA * torch.mean(torch.log(D_real_gauss + EPSILON) + torch.log(1-D_fake_gauss+EPSILON))
        D_loss.backward()
        D_optim.step()

        encoder.train()
        z_fake_gauss = encoder.forward(real_data_v)
        D_fake_gauss = discriminator.forward(z_fake_gauss)

        G_loss = -LAMBDA*torch.mean(torch.log(D_fake_gauss + EPSILON))
        G_loss.backward()
        Q_gen_optim.step()

        step += 1
        train_loss_recon += recon_loss.data[0]
        train_D_loss += D_loss.data[0]
        train_G_loss += G_loss.data[0]

        # if i % 100 == 0:
        #     print("Epoch: %d, Step: [%d/%d], Reconstruction Loss: %.4f, Discriminator Loss: %.4f, Generator Loss: %.4f" %
        #           (iteration + 1, step + 1, len(data_loader), recon_loss.data[0], D_loss.data[0], G_loss.data[0]))

    tmp = np.asarray([train_loss_recon, train_D_loss, train_G_loss]) / len(data_loader) / BATCH_SIZE
    print('[%.4f, %.4f, %.7f],' % (tmp[0], tmp[1], tmp[2]))

    encoder.eval()
    decoder.eval()
    discriminator.eval()
    tmp = evaluate(valid_loader)

    print("[%.4f,%.4f,%.7f]," % (tmp[0], tmp[1], tmp[2]))

    recon = inverse_sigmoid(decoder(encoder(F.sigmoid(fixed_x))))
    recover_sound(recon, iteration, 'wae')


