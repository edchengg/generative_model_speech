import time
import argparse
import numpy as np
import torchvision

from torchvision import datasets, transforms
import torch
import torch.utils.data
from torch.autograd import Variable
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from model.vae import *
from model.vae11 import *
from utils import *


parser = argparse.ArgumentParser(description='VAE speech reconstruction')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='initial learning rate')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='batch size')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=500,
                    help='upper epoch limit')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')
parser.add_argument('--zdim', type=int, default=128,
                    help='number of latent variables')
parser.add_argument('--Lambda', type=int, default=1,
                    help='lambda for KL')
parser.add_argument('--gamma', type=int, default=1,
                    help='gamma for reconstruction')
parser.add_argument('--logvar', type=float, default=0.1,
                    help='log variance')
parser.add_argument('--BCE', action='store_true',
                    help='cross entropy loss')
parser.add_argument('--dropout', type=float, default=0.1,
                    help='dropout rate')
parser.add_argument('--leakyrelu', type=float, default=0.2,
                    help='leaky relu rate')
parser.add_argument('--adam', action='store_false',
                    help='use adam')
parser.add_argument('--beta1', type=float, default=0.99,
                    help='beta1')
parser.add_argument('--beta2', type=float, default=0.95,
                    help='beta2')
parser.add_argument('--eps', type=float, default=1e-8,
                    help='epsilon')
parser.add_argument('--L2', type=float, default=1e-4,
                    help='L2 regularization')
parser.add_argument('--label', type=int, default=0,
                    help='plot figure label')
parser.add_argument('--spect', action='store_true',
                    help='use spectrogram as input 19 frame')
args = parser.parse_args()

############################################### Parameters #################################################
BATCH_SIZE = args.batch_size  # Batch size
ITERS = args.epochs # How many generator iterations to train for
# ADAM parameters
ADAM = args.adam
LEARNING_RATE = args.lr
L2 = args.L2
BETA1 = args.beta1
BETA2 = args.beta2
EPS = args.eps
CLIP = args.clip
# KL
LAMBDA = args.Lambda # KL
GAMMA = args.gamma #reconstruction
# Model Parameter
BCE = args.BCE
dropout = args.dropout
leakyrelu = args.leakyrelu
# Not fix/Fix latent variance
LOG_VAR_ = args.logvar # sigma
Z_DIM = args.zdim


FRAME = 19
SPECT = args.spect


torch.manual_seed(args.seed)

use_cuda = torch.cuda.is_available()


####################################### Load data ############################################
if SPECT:
    data_train = np.load('./data/train_log_spectrogram_19_128.npy')
    data_loader = torch.utils.data.DataLoader(data_train,
                                              batch_size=BATCH_SIZE,
                                              shuffle=True)

    data_dev = np.load('./data/dev_log_spectrogram_19_128.npy')
    dev_loader = torch.utils.data.DataLoader(data_dev,
                                             batch_size=BATCH_SIZE,
                                             shuffle=True)

    data_fix = np.load('./data/train_log_spectrogram_19_128.npy')
    fix_loader = torch.utils.data.DataLoader(data_fix,
                                             batch_size=1,
                                             shuffle=False)
else:
    data_train = np.load('./data/train_40_nolabel_' + str(FRAME)+'.npy')
    data_loader = torch.utils.data.DataLoader(data_train,
                                              batch_size=BATCH_SIZE,
                                              shuffle=True)

    data_dev = np.load('./data/dev_40_nolabel_' + str(FRAME)+'.npy')
    dev_loader = torch.utils.data.DataLoader(data_dev,
                                              batch_size=BATCH_SIZE,
                                              shuffle=True)

    data_fix = np.load('./data/train_40_nolabel_' + str(FRAME)+'.npy')
    fix_loader = torch.utils.data.DataLoader(data_fix,
                                              batch_size=1,
                                              shuffle=False)



iter_per_epoch = len(data_loader)

if LOG_VAR_ is None:
    model = VAE(Z_DIM, dropout=dropout, relu=leakyrelu, spectrogram=SPECT)
else:
    model = VAE(Z_DIM, LOG_VAR_, dropout=dropout, relu=leakyrelu, spectrogram=SPECT)

if use_cuda:
    model.cuda()

# Optimizers
if ADAM:
    optimizer = optim.Adam(model.parameters(),lr=LEARNING_RATE,betas=(BETA1,BETA2),eps=EPS,weight_decay=L2)
else:
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=BETA1, nesterov=True, weight_decay=L2)

print(model)

# fix sample in training set for debug
fix_iter = iter(fix_loader)
_= next(fix_iter)
_ = next(fix_iter)
fixed_x = next(fix_iter)
fixed_x = Variable(fixed_x).float()

if use_cuda:
    fixed_x = fixed_x.cuda()

if SPECT:
    plot_spectrogram(fixed_x.data.numpy(), 'rea','vae',FRAME,0)
else:
    recover_sound(fixed_x,'real','vae',FRAME, 0)


def evaluate(dev_data):
    loss_epoch = 0
    recon_loss_epoch = 0
    kl_loss_epoch = 0
    for i, data_i in enumerate(dev_data):

        data_i = Variable(data_i).float()

        if use_cuda:
            data_i = data_i.cuda()

        out, mu, log_var = model(data_i)
        # Compute reconstruction loss and kL divergence
        reconst_loss = F.mse_loss(out, data_i, size_average=False)
        kl_divergence = torch.sum(0.5 * (mu ** 2 + torch.exp(log_var) - log_var - 1))
        #kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        # Back propagation + Optimize
        total_loss = GAMMA*reconst_loss + LAMBDA*kl_divergence

        loss_epoch += total_loss.item()
        recon_loss_epoch += reconst_loss.item()
        kl_loss_epoch += kl_divergence.item()
    return [loss_epoch/len(dev_data)/BATCH_SIZE, recon_loss_epoch/len(dev_data)/BATCH_SIZE, \
           kl_loss_epoch/len(dev_data)/BATCH_SIZE]


for epoch in range(ITERS):
    model.train()
    loss_epoch = 0
    recon_loss_epoch = 0
    kl_loss_epoch = 0
    #TODO: optimizer wrapper for learning rate
    for i, data_i in enumerate(data_loader):

        data_i = Variable(data_i).float()
        if use_cuda:
            data_i = data_i.cuda()

        out, mu, log_var = model(data_i)

        # Compute reconstruction loss and kL divergence
        if BCE:
            reconst_loss = F.binary_cross_entropy(F.sigmoid(out), F.sigmoid(data_i), size_average=False)
        else:
            reconst_loss = F.mse_loss(out, data_i, size_average=False)
        kl_divergence = torch.sum(0.5 * (mu ** 2 + torch.exp(log_var) - log_var - 1))

        # Back propagation + Optimize
        total_loss = GAMMA * reconst_loss + LAMBDA * kl_divergence
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), CLIP)
        optimizer.step()

        loss_epoch += total_loss.item()
        recon_loss_epoch += reconst_loss.item()
        kl_loss_epoch += kl_divergence.item()
        if i % 100 == 0:
            print ("Epoch[%d/%d], Step [%d/%d], Total Loss: %.4f, "
                   "Reconst Loss: %.4f, KL Div: %.7f"
                   %(epoch+1, ITERS, i+1, iter_per_epoch, total_loss.data[0]/BATCH_SIZE,
                     reconst_loss.data[0]/BATCH_SIZE, kl_divergence.data[0]/BATCH_SIZE))

    tmp = np.asarray([loss_epoch, recon_loss_epoch, kl_loss_epoch])/len(data_loader)/BATCH_SIZE
    print("Epoch[%d/%d], Total Loss: %.4f, "
          "Reconst Loss: %.4f, KL Div: %.7f"
          % (epoch + 1, ITERS,tmp[0],tmp[1],tmp[2]))

    # Saved the reconstruct images

    model.eval()


    if epoch % 1 == 0:
        reconst, _, _ = model(fixed_x)
        if SPECT:
            plot_spectrogram(reconst.data.numpy(), epoch+1, 'vae', FRAME, 0)
        else:
            recover_sound(reconst, epoch+1, 'vae', FRAME, 0)
    loss = evaluate(dev_loader)
    print("DEV\n Epoch[%d/%d], Total Loss: %.4f, "
          "Reconst Loss: %.4f, KL Div: %.7f"
          % (epoch + 1, ITERS, loss[0],loss[1],loss[2]))

