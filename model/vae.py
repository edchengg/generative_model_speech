import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

DIM = 64 # Model dimensionality
OUTPUT_DIM = 40*9 # Number of pixels in Spectrum
Z_DIM = 128

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        '''
        kernel_size = (height, width)
        stride = (heigh, width)
        Conv1: filters = 64, filter size = 1 * F, stride = (1,1), tanh(), BatchNorm
        Cov2: filters = 128, filter size = 3 * 1, stride = (2,1), tanh(), BatchNorm
        Cov3: filters = 256, filter size = 3 * 1, stride = (2,1), tanh(), BatchNorm
        Fc1: Units = 512, tanh, BatchNorm
        Gauss: Units = 48
        
        1*(9*40) --> 64*(9*1) --> 128*(4*1) --> 256*(1*1) --> (512) --> (48)
        '''
        self.Conv_1 = nn.Sequential(
            nn.Conv2d(1, DIM, kernel_size=(1,40), stride=1, padding=0),
            nn.BatchNorm2d(DIM),
            nn.Tanh()
        )
        self.Conv_2 = nn.Sequential(
            nn.Conv2d(DIM, 2*DIM, kernel_size=(3,1), stride=(2,1), padding=0),
            nn.BatchNorm2d(2*DIM),
            nn.Tanh()
        )
        self.Conv_3 = nn.Sequential(
            nn.Conv2d(2*DIM, 4*DIM, kernel_size=(3,1), stride=(2,1), padding=0),
            nn.BatchNorm2d(4*DIM),
            nn.Tanh()
        )

        self.Fc1 =nn.Sequential(
            nn.Linear(4*DIM, 8*DIM),
            nn.BatchNorm1d(8*DIM),
            nn.Tanh()
        )
        self.mu = nn.Linear(8*DIM, Z_DIM)
        self.log_var = nn.Linear(8*DIM, Z_DIM)


    def forward(self, input):
        output = input.view(-1, 1, 9, 40)
        out = self.Conv_1(output)
        out = self.Conv_2(out)
        out = self.Conv_3(out)
        out = out.view(-1, 4*DIM)
        out = self.Fc1(out)
        mu, log_var = self.mu(out), self.log_var(out)
        return mu, log_var

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        '''
        Gauss: Units = 48
        Fc1: Units = 512, tanh, BatchNorm
        Cov3: filters = 256, filter size = 3 * 1, stride = (2,1), tanh(), BatchNorm
        Cov2: filters = 128, filter size = 3 * 1, stride = (2,1), tanh(), BatchNorm
        Conv1: filters = 64, filter size = 1 * F, stride = (1,1), tanh(), BatchNorm
        1*(48*1) --> 1*(512*1) --> 256*(1*1) --> 128*(4*1) --> 64*(9*1) --> 1*(9*40)
        '''
        self.Gauss = nn.Linear(Z_DIM, 8*DIM)

        self.Fc1 = nn.Sequential(
            nn.Linear(8*DIM, 4*DIM),
            nn.BatchNorm1d(4*DIM),
            nn.LeakyReLU(0.2)
        )

        self.Conv_3 = nn.Sequential(
            nn.ConvTranspose2d(4*DIM, 2*DIM, kernel_size=(3,1), stride=(2,1), output_padding=(1,0)),
            nn.BatchNorm2d(2*DIM),
            nn.LeakyReLU(0.2)
        )

        self.Conv_2 = nn.Sequential(
            nn.ConvTranspose2d(2*DIM, DIM, kernel_size=(3,1), stride=(2,1), padding=(0,0)),
            nn.BatchNorm2d(DIM),
            nn.LeakyReLU(0.2)
        )

        self.Conv_1 = nn.Sequential(
            nn.ConvTranspose2d(DIM*1, 1, kernel_size=(1,40), stride=1, padding=(0,0), output_padding=(0,0)),
            nn.BatchNorm2d(1),
            nn.Tanh()
        )


    def forward(self, input):
        output = self.Gauss(input)
        output = self.Fc1(output)
        output = output.view(-1, 4*DIM, 1, 1)
        output = self.Conv_3(output)
        output = self.Conv_2(output)
        output = self.Conv_1(output)
        output = output.view(-1, 9*40)
        return output

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        '''
        VAE:
        Input --> Encoder --> mu,log_var --> Decoder --> Output
        1 * (9*40) -->  Encoder --> 1 * 48 --> Decoder --> 1 * (9*40) 
        '''
        self.encoder = Encoder()
        self.decoder = Decoder()

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
            return eps.mul(std).add_(mu)
        else:
            return mu

    def sample(self, z):
        return self.decoder(z)


