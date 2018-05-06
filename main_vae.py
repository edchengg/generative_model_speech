import time

import matplotlib
matplotlib.use('Agg')
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

from utils import *

BATCH_SIZE = 128 # Batch size
ITERS = 20 # How many generator iterations to train for
# ADAM parameters
LEARNING_RATE = 1e-3
L2 = 1e-4
BETA1 = 0.99
BETA2 = 0.95
EPS = 1e-8
CLIP = 0.25
# KL
LAMBDA = 1 # KL
GAMMA = 1 #reconstruction
# Model Parameter
# Fix latent variance
LOG_VAR_ = 0.5 #sigma
Z_DIM = 48

torch.manual_seed(1)
use_cuda = torch.cuda.is_available()

data_train = np.load('./data/traindata_19_40_nolabel.npy')
data_loader = torch.utils.data.DataLoader(data_train,
                                          batch_size=BATCH_SIZE,
                                          shuffle=True)

data_dev = np.load('./data/devdata_19_40_nolabel.npy')
dev_loader = torch.utils.data.DataLoader(data_dev,
                                          batch_size=BATCH_SIZE,
                                          shuffle=True)

data_fix = np.load('./data/traindata_19_40_nolabel.npy')
fix_loader = torch.utils.data.DataLoader(data_fix,
                                          batch_size=1,
                                          shuffle=False)



# TODO: 1. How should I processing the data to use cross-entropy loss function
iter_per_epoch = len(data_loader)


model = VAE(LOG_VAR_, Z_DIM)

if use_cuda:
    model.cuda()

# Optimizers
optimizer = optim.Adam(model.parameters(),lr=LEARNING_RATE,betas=(BETA1,BETA2),eps=EPS,weight_decay=L2)

fix_iter = iter(fix_loader)
_= next(fix_iter)
fixed_x = next(fix_iter)
fixed_x = Variable(fixed_x).float()

if use_cuda:
    fixed_x = fixed_x.cuda()
recover_sound(fixed_x,'real')


def inverse_sigmoid(input):
    res =  -1*torch.log(1/input - 1)
    return res


def evaluate(dev_data):
    loss_epoch = 0
    recon_loss_epoch = 0
    kl_loss_epoch = 0
    for i, data_i in enumerate(dev_data):
        start_time = time.time()

        data_i = Variable(data_i).float()

        if use_cuda:
            data_i = data_i.cuda()
        #data_i = to_var(data_i.view(data_i.size(0),-1))
        out, mu, log_var = model(data_i)
        # Compute reconstruction loss and kL divergence
        reconst_loss = F.binary_cross_entropy(out, F.sigmoid(data_i), size_average=False)
        kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        # Back propagation + Optimize
        total_loss = GAMMA*reconst_loss + LAMBDA*kl_divergence
        #print(out)
        loss_epoch += total_loss
        recon_loss_epoch += reconst_loss
        kl_loss_epoch += kl_divergence
    return [loss_epoch/len(dev_data)/BATCH_SIZE, recon_loss_epoch/len(dev_data)/BATCH_SIZE, \
           kl_loss_epoch/len(dev_data)/BATCH_SIZE]

#train_loss = []
#dev_loss = []

for epoch in range(ITERS):
    model.train()
    loss_epoch = 0
    recon_loss_epoch = 0
    kl_loss_epoch = 0
    for i, data_i in enumerate(data_loader):
        start_time = time.time()

        data_i = Variable(data_i).float()

        ## sigmoid data_i
        #data_i = F.sigmoid(data_i)

        if use_cuda:
            data_i = data_i.cuda()
        #data_i = to_var(data_i.view(data_i.size(0),-1))
        out, mu, log_var = model(data_i)

        # Compute reconstruction loss and kL divergence
        reconst_loss = F.binary_cross_entropy(out, F.sigmoid(data_i), size_average=False)
        kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        # Back propagation + Optimize
        total_loss = GAMMA*reconst_loss + LAMBDA*kl_divergence
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), CLIP)
        optimizer.step()
        #print(out)
        loss_epoch += total_loss
        recon_loss_epoch += reconst_loss
        kl_loss_epoch += kl_divergence
        if i % 100 == 0:
            print ("Epoch[%d/%d], Step [%d/%d], Total Loss: %.4f, "
                   "Reconst Loss: %.4f, KL Div: %.7f"
                   %(epoch+1, ITERS, i+1, iter_per_epoch, total_loss.data[0]/BATCH_SIZE,
                     reconst_loss.data[0]/BATCH_SIZE, kl_divergence.data[0]/BATCH_SIZE))
    tmp = np.asarray([loss_epoch, recon_loss_epoch, kl_loss_epoch])/len(data_loader)/BATCH_SIZE
    print("Epoch[%d/%d], Total Loss: %.4f, "
          "Reconst Loss: %.4f, KL Div: %.7f"
          % (epoch + 1, ITERS,tmp[0],tmp[1],tmp[2]))
    #train_loss.append(tmp)

    #TODO: Test Development Set Loss, if Loss not decrease for 10 epoch stop training and save the model
    # Saved the reconstruct images
    model.eval()
    #print(fixed_x)
    #reconst,_,_ = model(F.sigmoid(fixed_x))
    reconst, _, _ = model(fixed_x)
    #reconst = inverse_sigmoid(reconst)
    #print(reconst)
    recover_sound(inverse_sigmoid(reconst), epoch+1)
    loss = evaluate(dev_loader)
    print("DEV\n Epoch[%d/%d], Total Loss: %.4f, "
          "Reconst Loss: %.4f, KL Div: %.7f"
          % (epoch + 1, ITERS, loss[0],loss[1],loss[2]))
    #dev_loss.append(loss)

