# Generative Models

## Description

This repository contains a pytorch implementation of the paper [Learning Latent Representations for Speech Generation and Transformation](https://arxiv.org/abs/1704.04222).



### Model 

<img src="https://github.com/edchengg/generative_model_speech/blob/master/figures/cnn_vae.png" width="600">


### Testing set reconstruction
<img src="https://github.com/edchengg/generative_model_speech/blob/master/figures/vae/sa2_spectrogram_80_128_recon.png" width="600">

<img src="https://github.com/edchengg/generative_model_speech/blob/master/figures/vae/sa2_waveform_80_128.png" width="600">

Input sound example: [Real](https://drive.google.com/open?id=1T3DtVVw87XwBFYkIsMbu_NaR4WJbfNNV)

Reconstructed sound example: [Reconstruction](https://drive.google.com/open?id=1MKPfCWWNlJ0dGXa5p4c9UDrEa6WTvn1r)

### Conversion Male to Female
<img src="https://github.com/edchengg/generative_model_speech/blob/master/figures/vae/Conversion_M2F.png" width="600">
Input Male sound example: [Male sound](https://drive.google.com/open?id=1wjHh6YXSDWsE_r_ZJ2aoTRO2GlCOwSg_)

Output Female sound example: [Modified sound](https://drive.google.com/open?id=1ILwvU30PX4qo3jyy4g4sJdT0mArZAhgs)

### Conversion Female to Male
<img src="https://github.com/edchengg/generative_model_speech/blob/master/figures/vae/sx197_sx366_spectrogram_80_128_f2m.png" width="600">

### Conversion Male to Female
<img src="https://github.com/edchengg/generative_model_speech/blob/master/figures/vae/sx197_sx366_spectrogram_80_128_m2f.png" width="600">

### T-SNe

#### Gender

<img src="https://github.com/edchengg/generative_model_speech/blob/master/figures/vae/tsne_gender.png" width="400">

#### Dialect

<img src="https://github.com/edchengg/generative_model_speech/blob/master/figures/vae/tsne_dr.png" width="400">

### Generation
<img src="https://github.com/edchengg/generative_model_speech/blob/master/figures/vae/generation.png" width="600">


## Tuning
### Training Loss:
<img src="https://github.com/edchengg/generative_model_speech/blob/master/figures/diff_loss/loss_plot.png" width="400">


## Data
### Timit
Train/Dev/Test: 3696/400/192

The input data is Log Mel filter banks with number of filters = [40/64/80] and the window size is 19.
The input data is then converted to size = [1,760/1216/1520].

## Run
```
$ python main_vae.py --help
usage: main_vae.py [-h] [--lr LR] [--batch_size N] [--clip CLIP]
                   [--epochs EPOCHS] [--seed SEED] [--zdim ZDIM]
                   [--Lambda LAMBDA] [--gamma GAMMA] [--logvar LOGVAR] [--BCE]
                   [--dropout DROPOUT] [--leakyrelu LEAKYRELU] [--adam]
                   [--beta1 BETA1] [--beta2 BETA2] [--eps EPS] [--L2 L2]
                   [--ID ID] [--nfilter NFILTER]
                   
VAE speech reconstruction

optional arguments:
  -h, --help            show this help message and exit
  --lr LR               initial learning rate
  --batch_size N        batch size
  --clip CLIP           gradient clipping
  --epochs EPOCHS       upper epoch limit
  --seed SEED           random seed
  --zdim ZDIM           number of latent variables
  --Lambda LAMBDA       lambda for KL
  --gamma GAMMA         gamma for reconstruction
  --logvar LOGVAR       log variance
  --BCE                 cross entropy loss
  --dropout DROPOUT     dropout rate
  --leakyrelu LEAKYRELU
                        leaky relu rate
  --adam                use adam
  --beta1 BETA1         beta1
  --beta2 BETA2         beta2
  --eps EPS             epsilon
  --L2 L2               L2 regularization
  --ID ID               plot figure label
  --nfilter NFILTER     number of filters in mel filter bank
```
