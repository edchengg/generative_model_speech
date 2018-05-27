# Generative Models

## Description

This repository contains a pytorch implementation of the paper [Learning Latent Representations for Speech Generation and Transformation](https://arxiv.org/abs/1704.04222).



### Model 

<img src="https://github.com/edchengg/generative_model_speech/blob/master/figures/cnn_vae.png" width="600">

### Training Set reconstruction
<img src="https://github.com/edchengg/generative_model_speech/blob/master/figures/vae/training.png" width="600">

### Validation set reconstruction
<img src="https://github.com/edchengg/generative_model_speech/blob/master/figures/vae/validation.png" width="600">

### Testing set reconstruction
<img src="https://github.com/edchengg/generative_model_speech/blob/master/figures/vae/testing.png" width="600">

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
python main_vae.py --epoch 400 --gamma 3 --dropout 0 --leakyrelu 0.4 --ID 1 --lr 1e-3 --zdim 128 --logvar 0 --nfilter 40
```
