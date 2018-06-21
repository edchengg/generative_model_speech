import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

DIM = 64  # Model dimensionality
OUTPUT_DIM = 40 * 9  # Number of pixels in Spectrum


## Convolutional Variational Autoencoder

class Encoder(nn.Module):
    def __init__(self, Z_DIM, log_var_=None, dropout=0.2, relu=0.1, n_filters=40):
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
        if relu == 0:
            activation = nn.Tanh()
        else:
            activation = nn.LeakyReLU(relu)

        self.F = n_filters

        self.Conv_1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(1, self.F), stride=1, padding=0),
            nn.BatchNorm2d(64)
        )
        self.Conv_1_gate = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(1, self.F), stride=1, padding=0),
            nn.BatchNorm2d(64)
        )

        self.Conv_2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 1), stride=(2, 1), padding=0),
            nn.BatchNorm2d(128)
        )
        self.Conv_2_gate = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 1), stride=(2, 1), padding=0),
            nn.BatchNorm2d(128)
        )
        self.Conv_3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3, 1), stride=(2, 1), padding=0),
            nn.BatchNorm2d(256)
        )
        self.Conv_3_gate = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3, 1), stride=(2, 1), padding=0),
            nn.BatchNorm2d(256)
        )

        self.Fc1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            activation
        )

        self.mu = nn.Linear(512, Z_DIM)
        self.log_var = nn.Linear(512, Z_DIM)

    def forward(self, input):
        output = input.view(-1, 1, 19, self.F)
        A = self.Conv_1(output)
        B = self.Conv_1_gate(output)
        out = A * F.sigmoid(B)
        A = self.Conv_2(out)
        B = self.Conv_2_gate(out)
        out = A * F.sigmoid(B)
        A = self.Conv_3(out)
        B = self.Conv_3_gate(out)
        out = A * F.sigmoid(B)
        out = out.view(-1, 1024)
        out = self.Fc1(out)
        mu = self.mu(out)
        log_var = self.log_var(out)
        return mu, log_var


class Decoder(nn.Module):
    def __init__(self, Z_DIM, dropout=0.2, relu=0.1, n_filters=40):
        super(Decoder, self).__init__()
        '''
        Gauss: Units = 48
        Fc1: Units = 512, tanh, BatchNorm
        Cov3: filters = 256, filter size = 3 * 1, stride = (2,1), tanh(), BatchNorm
        Cov2: filters = 128, filter size = 3 * 1, stride = (2,1), tanh(), BatchNorm
        Conv1: filters = 64, filter size = 1 * F, stride = (1,1), tanh(), BatchNorm
        1*(128*1) --> 1*(512*1) --> 256*(4*1) --> 128*(9*1) --> 64*(19*1) --> 1*(19*40)
        '''
        if relu == 0:
            activation = nn.Tanh()
        else:
            activation = nn.LeakyReLU(relu)

        self.F = n_filters

        self.Gauss = nn.Linear(Z_DIM, 512)

        self.Fc1 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            activation
        )
        # self.deconv3 = nn.ConvTranspose2d(4*DIM, 2*DIM, kernel_size=(3,1),stride=(2,1))

        self.Conv_3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=(3, 1), stride=(2, 1)),
            nn.BatchNorm2d(128)
        )
        self.Conv_3_gate = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=(3, 1), stride=(2, 1)),
            nn.BatchNorm2d(128)
        )

        self.Conv_2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=(3, 1), stride=(2, 1)),
            nn.BatchNorm2d(64)
        )

        self.Conv_2_gate = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=(3, 1), stride=(2, 1)),
            nn.BatchNorm2d(64)
        )

        self.Conv_1 = nn.Sequential(
            nn.ConvTranspose2d(64, 1, kernel_size=(1, self.F), stride=1),
            nn.BatchNorm2d(1)
        )
        self.Conv_1_gate = nn.Sequential(
            nn.ConvTranspose2d(64, 1, kernel_size=(1, self.F), stride=1),
            nn.BatchNorm2d(1)
        )

    def forward(self, input):
        output = self.Gauss(input)
        output = self.Fc1(output)
        output = output.view(-1, 256, 4, 1)
        A = self.Conv_3(output)
        B = self.Conv_3_gate(output)
        output = A * F.sigmoid(B)
        A = self.Conv_2(output)
        B = self.Conv_2_gate(output)
        output = A * F.sigmoid(B)
        A = self.Conv_1(output)
        B = self.Conv_1_gate(output)
        output = A * F.sigmoid(B)
        output = output.view(-1, 19 * self.F)
        return output


class VAE(nn.Module):
    def __init__(self, Z_DIM, log_var_=None, dropout=0.2, relu=0.1, n_filters=40):
        super(VAE, self).__init__()

        '''
        VAE:
        Input --> Encoder --> mu,log_var --> Decoder --> Output
        1 * (9*40) -->  Encoder --> 1 * 48 --> Decoder --> 1 * (9*40) 
        '''
        self.encoder = Encoder(Z_DIM, log_var_, dropout=dropout, relu=relu, n_filters=n_filters)
        self.decoder = Decoder(Z_DIM, dropout=dropout, relu=relu, n_filters=n_filters)

    def forward(self, input):
        mu, log_var = self.encoder(input)
        # reparam
        z = self.reparameterize(mu.float(), log_var.float())
        output = self.decoder(z)
        return output, mu, log_var

    def reparameterize(self, mu, log_var):
        if self.training:
            std = log_var.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            if torch.cuda.is_available():
                eps = eps.cuda()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def sample(self, z):
        return self.decoder(z)