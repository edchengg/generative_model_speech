import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

DIM = 64 # Model dimensionality
OUTPUT_DIM = 760 # Number of pixels in MNIST (28*28)

class Decoder(nn.Module):
    def __init__(self, Z_DIM):
        super(Decoder, self).__init__()
        '''
        Gauss: Units = 48
        Fc1: Units = 512, tanh, BatchNorm
        Cov3: filters = 256, filter size = 3 * 1, stride = (2,1), tanh(), BatchNorm
        Cov2: filters = 128, filter size = 3 * 1, stride = (2,1), tanh(), BatchNorm
        Conv1: filters = 64, filter size = 1 * F, stride = (1,1), tanh(), BatchNorm
        1*(128*1) --> 1*(512*1) --> 256*(4*1) --> 128*(9*1) --> 64*(19*1) --> 1*(19*40)
        '''
        self.Gauss = nn.Linear(Z_DIM, 512)

        self.Fc1 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.Tanh()
        )
        #self.deconv3 = nn.ConvTranspose2d(4*DIM, 2*DIM, kernel_size=(3,1),stride=(2,1))

        self.Conv_3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=(3,1), stride=(2,1)),
            nn.BatchNorm2d(128),
            nn.Tanh()
        )

        self.Conv_2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=(3,1), stride=(2,1)),
            nn.BatchNorm2d(64),
            nn.Tanh()
        )


        self.Conv_1 = nn.Sequential(
            nn.ConvTranspose2d(64, 1, kernel_size=(1,40), stride=1),
            nn.BatchNorm2d(1),
            nn.Tanh()
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        output = self.Gauss(input)
        output = self.Fc1(output)
        output = output.view(-1, 256, 4, 1)
        output = self.Conv_3(output)
        output = self.Conv_2(output)
        output = self.Conv_1(output)
        output = output.view(-1, 19*40)
        return self.sigmoid(output)

class Discriminator(nn.Module):
    def __init__(self, Z_DIM):
        super(Discriminator, self).__init__()
        '''
        Gauss: Units = 48
        Fc1: Units = 512, tanh, BatchNorm
        Cov3: filters = 256, filter size = 3 * 1, stride = (2,1), tanh(), BatchNorm
        Cov2: filters = 128, filter size = 3 * 1, stride = (2,1), tanh(), BatchNorm
        Conv1: filters = 64, filter size = 1 * F, stride = (1,1), tanh(), BatchNorm
        1*(128*1) --> 1*(512*1) --> 256*(4*1) --> 128*(9*1) --> 64*(19*1) --> 1*(19*40)
        '''
        self.Gauss = nn.Linear(Z_DIM, 512)

        self.Fc1 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.Tanh()
        )
        #self.deconv3 = nn.ConvTranspose2d(4*DIM, 2*DIM, kernel_size=(3,1),stride=(2,1))

        self.Conv_3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=(3,1), stride=(2,1)),
            nn.BatchNorm2d(128),
            nn.Tanh()
        )

        self.Conv_2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=(3,1), stride=(2,1)),
            nn.BatchNorm2d(64),
            nn.Tanh()
        )


        self.Conv_1 = nn.Sequential(
            nn.ConvTranspose2d(64, 1, kernel_size=(1,40), stride=1),
            nn.BatchNorm2d(1),
            nn.Tanh()
        )

        self.Fc2 = nn.Sequential(
            nn.Linear(19*40, 256),
            nn.BatchNorm1d(256),
            nn.Tanh()
        )

        self.Linear1 = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        output = self.Gauss(input)
        output = self.Fc1(output)
        output = output.view(-1, 256, 4, 1)
        output = self.Conv_3(output)
        output = self.Conv_2(output)
        output = self.Conv_1(output)
        output = output.view(-1, 19*40)
        output = self.Fc2(output)
        output = self.Linear1(output)
        return self.sigmoid(output)

class Encoder(nn.Module):
    def __init__(self, Z_DIM):
        super(Encoder, self).__init__()
        '''
        kernel_size = (height, width)
        stride = (heigh, width)
        Conv1: filters = 64, filter size = 1 * F, stride = (1,1), tanh(), BatchNorm
        Cov2: filters = 128, filter size = 3 * 1, stride = (2,1), tanh(), BatchNorm
        Cov3: filters = 256, filter size = 3 * 1, stride = (2,1), tanh(), BatchNorm
        Fc1: Units = 512, tanh, BatchNorm
        Gauss: Units = 128

        1*(19*40) --> 64*(19*1) --> 128*(9*1) --> 256*(4*1) --> (512) --> (128)
        '''
        self.Conv_1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(1, 40), stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.Tanh()
        )
        self.Conv_2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 1), stride=(2, 1), padding=0),
            nn.BatchNorm2d(128),
            nn.Tanh()
        )
        self.Conv_3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3, 1), stride=(2, 1), padding=0),
            nn.BatchNorm2d(256),
            nn.Tanh()
        )

        self.Fc1 = nn.Sequential(
            nn.Linear(1024, 512),  #
            nn.BatchNorm1d(512),
            nn.Tanh()
        )
        self.mu = nn.Linear(512, Z_DIM)
        self.sigmoid = nn.Sigmoid()
        # self.log_var = nn.Linear(8*DIM, Z_DIM)

    def forward(self, input):
        output = input.view(-1, 1, 19, 40)
        output = self.sigmoid(output)
        out = self.Conv_1(output)
        out = self.Conv_2(out)
        out = self.Conv_3(out)
        out = out.view(-1, 1024)
        out = self.Fc1(out)
        mu = self.mu(out)
        return mu

