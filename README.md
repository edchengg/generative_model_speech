# Generative Models

## Description

This repository contains a group of work for speech generation project.

## TODO List

- [x] Data processing
- [x] WAE
- [x] WGAN-GP
- [x] VAE+GAN


## WGAN-GP(Wasserstein GAN-gradient penalty)

<img src="https://github.com/edchengg/generative_model_speech/blob/master/figures/wgan.png" width="800">

### Results
![Alt Text](https://github.com/edchengg/generative_model_speech/blob/master/figures/gan_result.gif)

[Improved Training of Wasserstein GANs](https://arxiv.org/pdf/1704.00028.pdf)

## WAE(Wasserstein AutoEncoder)

<img src="https://github.com/edchengg/generative_model_speech/blob/master/figures/wae_gan.png" width="800">

### Results
![Alt Text](https://github.com/edchengg/generative_model_speech/blob/master/figures/wae_result.gif)

[Wasserstein Auto-Encoders](https://arxiv.org/abs/1711.01558)


## Data

Logmel_normalized_48Label_Framealigned.mat


|train, dev, test| trainlen, devlen, testlen|train_dict,dev_dict,test_dict|
| :-------------: |:-------------:| :-----:|
| log filterbank features of the train, dev and test splits respectively | the length of each utterance in train, dev and test splits respectively |  per-frame aligned labels(0-48) |
| Number: (1124823, 122487, 57919)| Number: (3696, 400, 192) | kth frame of 'train' --> kth entry of 'train_dict' --> label. (sil = silence)|

After data processing (frame-window = 9):

|train_data| dev_data|test_data|
| :-------------: |:-------------:| :-----:|
|(3696,x,9,121)|(400,x,9,121)|(192,x,9,121)|
|x: number of (9,121) depends on frame-length|
|the last column is the label:train_data[0][0][:,-1]|
|[37,37,37,33,33,5,5,5,5]|[37,37,37,7,7,7,34,34,34]|[37,37,37,43,43,43,21,21,21]|

Transfer for batch training:

(3696,x,8,121) --> (y,$8 * 120$) [extract the last one(label)]

## Phone label:
|Phone|label|Phone|label|Phone|label|Phone|label|
| :--:| :--:| :--:| :--:| :--:| :--:| :--:| :--:|
|aa|0|ae|1|ah|2|ao|3|
|aw|4|ax|5|ay|6|b|7|
|ch|8|d|9|dh|10|dx|11|
|eh|12|el|13|en|14|epi|15|
|er|16|ey|17|f|18|g|19|
|hh|20|ih|21|ix|22|iy|23|
|jh|24|k|25|l|26|m|27|
|n|28|ng|29|ow|30|oy|31|
|p|32|q|33|r|34|s|35|
|sh|36|sil|37|t|38|th|39|
|cl|40|uh|41|uw|42|v|43|
|vcl|44|w|45|y|46|z|47|
|zh|48|






