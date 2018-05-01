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
ITERS = 200000 # How many generator iterations to train for
# ADAM parameters
LEARNING_RATE = 1e-3
L2 = 1e-4
BETA1 = 0.99
BETA2 = 0.95
EPS = 1e-8
# KL
LAMBDA = 1
GAMMA = 1

torch.manual_seed(1)
use_cuda = torch.cuda.is_available()

data_test = np.load('./data/train_data_40_nolabel.npy')
data_loader = torch.utils.data.DataLoader(data_test,
                                          batch_size=BATCH_SIZE,
                                          shuffle=True)

data_dev = np.load('./data/train_data_40_nolabel.npy')
dev_loader = torch.utils.data.DataLoader(data_test,
                                          batch_size=1,
                                          shuffle=True)
# Test structure on MNIST--> works
#dataset = datasets.MNIST(root='./data',
#                         train=True,
#                         transform=transforms.ToTensor(),
#                         download=True)

# Data loader
#data_loader = torch.utils.data.DataLoader(dataset=dataset,
#                                          batch_size=100,
#                                          shuffle=True)

iter_per_epoch = len(data_loader)


model = VAE()

if use_cuda:
    model.cuda()

# Optimizers
optimizer = optim.Adam(model.parameters(),lr=LEARNING_RATE,betas=(BETA1,BETA2),eps=EPS,weight_decay=L2)

dev_iter = iter(dev_loader)

# fixed inputs for debugging
#array([ 37.,  37.,  37.,  33.,  33.,   5.,   5.,   5.,   5.])
_ = next(dev_iter)
fixed_x = next(dev_iter)
fixed_x = Variable(fixed_x).float()
#def to_var(x):
#    if torch.cuda.is_available():
#        x = x.cuda()
#    return Variable(x)

#iter_per_epoch = len(data_loader)
#data_iter = iter(data_loader)
#fixed_z = to_var(torch.randn(100, 20))
#fixed_x, _ = next(data_iter)
#torchvision.utils.save_image(fixed_x.cpu(), './data/real_images.png')
#fixed_x = to_var(fixed_x.view(fixed_x.size(0), -1))


if use_cuda:
    fixed_x = fixed_x.cuda()
recover_sound(fixed_x,'real')

for epoch in range(ITERS):
    model.train()
    for i, data_i in enumerate(data_loader):
        start_time = time.time()

        data_i = Variable(data_i).float()
        if use_cuda:
            data_i = data_i.cuda()
        #data_i = to_var(data_i.view(data_i.size(0),-1))
        out, mu, log_var = model(data_i)

        # Compute reconstruction loss and kL divergence
        reconst_loss = F.mse_loss(out, data_i, size_average=False)
        kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        # Back propagation + Optimize
        total_loss = GAMMA*reconst_loss + LAMBDA*kl_divergence
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print ("Epoch[%d/%d], Step [%d/%d], Total Loss: %.4f, "
                   "Reconst Loss: %.4f, KL Div: %.7f"
                   %(epoch+1, ITERS, i+1, iter_per_epoch, total_loss.data[0]/BATCH_SIZE,
                     reconst_loss.data[0]/BATCH_SIZE, kl_divergence.data[0]/BATCH_SIZE))

    #TODO: Test Development Set Loss, if Loss not decrease for 10 epoch stop training and save the model
    # Saved the reconstruct images
    model.eval()
    reconst,_,_ = model(fixed_x)
    recover_sound(reconst, epoch+1)

            # Save the reconstructed images
    #reconst_images, _, _ = model(fixed_x)
    #reconst_images = reconst_images.view(reconst_images.size(0), 1, 28, 28)
    #torchvision.utils.save_image(reconst_images.data.cpu(),
    #                             './data/reconst_images_%d.png' % (epoch + 1))