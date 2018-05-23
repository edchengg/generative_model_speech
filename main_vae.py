import argparse
import os
import torch.utils.data
from model.vae import *
from utils import *
from torch.optim.lr_scheduler import *

parser = argparse.ArgumentParser(description='VAE speech reconstruction')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='initial learning rate')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='batch size')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=400,
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
parser.add_argument('--dropout', type=float, default=0,
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
parser.add_argument('--ID', type=int, default=0,
                    help='plot figure label')
parser.add_argument('--spect', action='store_true',
                    help='use spectrogram as input 19 frame')
args = parser.parse_args()

############################################### Parameters #################################################

BATCH_SIZE = args.batch_size  # Batch size
ITERS = args.epochs # How many generator iterations to train for

# Optimizer parameters
ADAM = args.adam
LEARNING_RATE = args.lr
L2 = args.L2
BETA1 = args.beta1
BETA2 = args.beta2
EPS = args.eps
CLIP = args.clip

# Loss function parameters
LAMBDA = args.Lambda # KL
GAMMA = args.gamma #reconstruction
BCE = args.BCE #use cross entropy

# Model parameters
dropout = args.dropout
leakyrelu = args.leakyrelu
if args.logvar == 0:
    LOG_VAR_ = None # None for unfix
else:
    LOG_VAR_ = args.logvar
Z_DIM = args.zdim


FRAME = 19
SPECT = args.spect
ID = args.ID

torch.manual_seed(args.seed)

use_cuda = torch.cuda.is_available()
if use_cuda:
    torch.cuda.manual_seed_all(args.seed)

####################################### Load data ############################################

if SPECT:
    data_train = np.load('./data/train_8_19_unnormalized.npy')
    train_loader = torch.utils.data.DataLoader(data_train,
                                              batch_size=BATCH_SIZE,
                                              shuffle=True)

    data_dev = np.load('./data/dev_8_19_unnormalized.npy')
    dev_loader = torch.utils.data.DataLoader(data_dev,
                                             batch_size=BATCH_SIZE,
                                             shuffle=True)
else:
    data_train = np.load('./data/train_unnormalized.npy')
    train_loader = torch.utils.data.DataLoader(data_train,
                                              batch_size=BATCH_SIZE,
                                              shuffle=True)

    data_dev = np.load('./data/dev_unnormalized.npy')
    dev_loader = torch.utils.data.DataLoader(data_dev,
                                              batch_size=BATCH_SIZE,
                                              shuffle=True)



if LOG_VAR_ is None:
    model = VAE(Z_DIM,
                dropout=dropout,
                relu=leakyrelu,
                spectrogram=SPECT)
else:
    model = VAE(Z_DIM,
                LOG_VAR_,
                dropout=dropout,
                relu=leakyrelu,
                spectrogram=SPECT)

if use_cuda:
    model.cuda()

# Optimizers
if ADAM:
    optimizer = optim.Adam(model.parameters(),
                           lr=LEARNING_RATE,
                           betas=(BETA1,BETA2),
                           eps=EPS,
                           weight_decay=L2)
else:
    optimizer = optim.SGD(model.parameters(),
                          lr=LEARNING_RATE,
                          momentum=BETA1,
                          nesterov=True,
                          weight_decay=L2)

print(model)

scheduler = StepLR(optimizer, step_size=100, gamma=0.5)


def evaluate(dev_data):
    model.eval()
    loss_epoch = 0
    recon_loss_epoch = 0
    kl_loss_epoch = 0

    for i, data_i in enumerate(dev_data):
        data_i =data_i.float()
        if use_cuda:
            data_i = data_i.cuda()
        out, mu, log_var = model(data_i)

        # Compute reconstruction loss and kL divergence
        if BCE:
            reconst_loss = F.binary_cross_entropy(F.sigmoid(out), F.sigmoid(data_i), size_average=False)
        else:
            reconst_loss = F.mse_loss(out, data_i, size_average=False)
        kl_divergence = torch.sum(0.5 * (mu ** 2 + torch.exp(log_var) - log_var - 1))
        total_loss = GAMMA*reconst_loss + LAMBDA*kl_divergence

        loss_epoch += total_loss.item()
        recon_loss_epoch += reconst_loss.item()
        kl_loss_epoch += kl_divergence.item()

    return [loss_epoch/len(dev_data)/BATCH_SIZE,
            recon_loss_epoch/len(dev_data)/BATCH_SIZE,
           kl_loss_epoch/len(dev_data)/BATCH_SIZE]

def train(train_data):
    scheduler.step()
    model.train()
    loss_epoch = 0
    recon_loss_epoch = 0
    kl_loss_epoch = 0
    for i, data_i in enumerate(train_data):
        data_i = data_i.float()
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        optimizer.step()

        loss_epoch += total_loss.item()
        recon_loss_epoch += reconst_loss.item()
        kl_loss_epoch += kl_divergence.item()

def main():
    train_loss_plot = []
    dev_loss_plot = []

    if not os.path.exists('save/'):
        os.makedirs('save/')

    best_loss = float("inf")

    for epoch in range(ITERS):

        # Train
        train(train_loader)
        # Evaluation
        train_loss = evaluate(train_loader)
        dev_loss = evaluate(dev_loader)

        train_loss_plot.append(train_loss)
        dev_loss_plot.append(dev_loss)

        print("Train: Epoch[%d/%d], Total Loss: %.4f, "
              "Reconst Loss: %.4f, KL Div: %.7f"
              % (epoch + 1, ITERS, train_loss[0],train_loss[1],train_loss[2]))

        print("Dev: Epoch[%d/%d], Total Loss: %.4f, "
              "Reconst Loss: %.4f, KL Div: %.7f"
              % (epoch + 1, ITERS, dev_loss[0],dev_loss[1],dev_loss[2]))

        # save the model
        if dev_loss[0] < best_loss:
            torch.save(model.state_dict(), 'save/best_model_' + str(ID) + '.pt')
            best_loss = dev_loss[0]

    train_loss_plot = np.vstack(train_loss_plot)
    dev_loss_plot = np.vstack(dev_loss_plot)
    plot_loss(train_loss_plot, dev_loss_plot, ID)
    np.save('./loss/train_loss_'+str(ID), train_loss_plot)
    np.save('./loss/dev_loss_'+str(ID), dev_loss_plot)


if __name__ == '__main__':
        main()