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
    def __init__(self, Z_DIM=64, log_var_=None, dropout=0.2, relu=0.1, n_filters=40):
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

        self.F = n_filters

        self.lstm = nn.LSTM(input_size=self.F, hidden_size=512, num_layers=2)

        self.mu = nn.Linear(512*2, Z_DIM)
        self.log_var = nn.Linear(512 * 2, Z_DIM)

    def forward(self, input):
        output = input.view(19, -1, self.F)
        _, (output, _) = self.lstm(output)
        output = output.view(-1, 512 * 2)
        mu = self.mu(output)
        log_var = self.log_var(output)
        return mu, log_var



class Decoder(nn.Module):
    def __init__(self, Z_DIM=64, dropout=0.2, relu=0.1, n_filters=40):
        super(Decoder, self).__init__()
        '''
        Gauss: Units = 48
        Fc1: Units = 512, tanh, BatchNorm
        Cov3: filters = 256, filter size = 3 * 1, stride = (2,1), tanh(), BatchNorm
        Cov2: filters = 128, filter size = 3 * 1, stride = (2,1), tanh(), BatchNorm
        Conv1: filters = 64, filter size = 1 * F, stride = (1,1), tanh(), BatchNorm
        1*(128*1) --> 1*(512*1) --> 256*(4*1) --> 128*(9*1) --> 64*(19*1) --> 1*(19*40)
        '''

        self.F = n_filters

        self.lstm = nn.LSTM(input_size = Z_DIM, hidden_size=512, num_layers=2)

        self.fc = nn.Linear(512, n_filters)

    def forward(self, input):
        inputs = input.unsqueeze(1)
        #print(inputs.size())
        inputs = inputs.repeat(1, 19, 1)
        #print(inputs.size())
        inputs = inputs.permute(1,0,2)
        #print(inputs.size())
        out, _ = self.lstm(inputs)
        #print(out.size())
        out = out.view(-1, 19, 512)
        out = self.fc(out)
        output = out.view(-1, 19 * self.F)
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
