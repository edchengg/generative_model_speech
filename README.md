# Generation Model in Speech

## Description

This repository contains a group of work for speech generation project.

## TODO List

- [x] Data processing
- [ ] VAE
- [ ] GAN
- [ ] VAE+GAN

## Data

Logmel_normalized_48Label_Framealigned.mat


|train, dev, test| trainlen, devlen, testlen|train_dict,dev_dict,test_dict|
| :-------------: |:-------------:| :-----:|
| log filterbank features of the train, dev and test splits respectively | the length of each utterance in train, dev and test splits respectively |  per-frame aligned labels(0-48) |
| Number: (1124823, 122487, 57919)| Number: (3696, 400, 192) | kth frame of 'train' --> kth entry of 'train_dict' --> label. (sil = silence)|

After data processing:

|train_data| dev_data|test_data|
| :-------------: |:-------------:| :-----:|
|(3696,9,121)|(400,9,121)|(192,9,121)|
|e.g., the last column is the label:train_data[0][0][:,-1] = [37,37,37,33,33,5,5,5,5]|||




