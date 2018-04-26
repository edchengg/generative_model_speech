import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

DIM = 64 # Model dimensionality
OUTPUT_DIM = 1080 # Number of pixels in MNIST (28*28)
Z_DIM = 128

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        preprocess = nn.Sequential(
            nn.Linear(Z_DIM, 4*4*4*DIM),
            nn.ReLU(True),
        )
        block1 = nn.Sequential(
                                # in_channels,out_channels,kernel_size
            nn.ConvTranspose2d(4*DIM, 2*DIM, 5),
            nn.ReLU(True),
        )
        block2 = nn.Sequential(
            nn.ConvTranspose2d(2*DIM, DIM, 5),
            nn.ReLU(True),
        )

        #Input: (N, Cin, Hin, Win)(N, Cin, Hin, Win)

        #Output: (N, Cout, Hout, Wout)(N, Cout, Hout, Wout)

        #Hout = (Hin−1)∗stride[0]−2∗padding[0] + kernel_size[0] + output_padding[0]
        #Wout = (Win−1)∗stride[1]−2∗padding[1] + kernel_size[1]

        deconv_out = nn.ConvTranspose2d(DIM, 1, kernel_size=(3,2), stride=(3,3), padding=(4,0),
                                        output_padding=(2,1))

        self.block1 = block1
        self.block2 = block2
        self.deconv_out = deconv_out
        self.preprocess = preprocess
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        output = self.preprocess(input)
        output = output.view(-1, 4*DIM, 4, 4)
        output = self.block1(output)

        #print(output.size())
        output = self.block2(output)
        #print(output.size())
        output = self.deconv_out(output)
        #print(output.size())
        output = self.sigmoid(output)

        return output.view(-1, OUTPUT_DIM)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        preprocess = nn.Sequential(
            nn.Linear(128, 4*4*4*DIM),
            nn.ReLU(True),
        )
        block1 = nn.Sequential(
                        # in_channels,out_channels,kernel_size
            nn.ConvTranspose2d(4*DIM, 2*DIM, 5),
            nn.ReLU(True),
        )
        block2 = nn.Sequential(
            nn.ConvTranspose2d(2*DIM, DIM, 5),
            nn.ReLU(True),
        )

        #Input: (N, Cin, Hin, Win)(N, Cin, Hin, Win)

        #Output: (N, Cout, Hout, Wout)(N, Cout, Hout, Wout)

        #Hout = (Hin−1)∗stride[0]−2∗padding[0] + kernel_size[0] + output_padding[0]
        #Wout = (Win−1)∗stride[1]−2∗padding[1] + kernel_size[1]

        deconv_out = nn.ConvTranspose2d(DIM, 1, kernel_size=(3,2), stride=(3,3), padding=(4,0),
                                        output_padding=(2,1))

        self.fc = nn.Linear(30*36,1)
        self.block1 = block1
        self.block2 = block2
        self.deconv_out = deconv_out
        self.preprocess = preprocess
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        output = self.preprocess(input)
        output = output.view(-1, 4*DIM, 4, 4)
        output = self.block1(output)

        #print(output.size())
        output = self.block2(output)
        #print(output.size())
        output = self.deconv_out(output)
        #print(output.size())
        output = output.view(output.size()[0],-1)
        #print(output.size())
        output = self.fc(output)
        output = F.relu(output)
        output = self.sigmoid(output)

        return output


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(1, DIM, 5, stride=2, padding=2),
            nn.ReLU(True)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(DIM, 2*DIM, 5, stride=2, padding=2),
            nn.ReLU(True)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(2*DIM, 4*DIM, 5, stride=2, padding=2),
            nn.ReLU(True)
        )


        #print(self.main)
        #self.output = nn.Linear(4*4*5*DIM, 1)
        #print(self.output)
        self.output = nn.Linear(4*4*5*DIM,Z_DIM)

    def forward(self, input):
        input = input.view(-1, 1, 30, 36)
        out = self.block1(input)
        #print(out.size())
        out = self.block2(out)
        #print(out.size())
        out = self.block3(out)
        #print(out.size())
        out = out.view(-1, 256*4*5)
        out = self.output(out)
        return out

