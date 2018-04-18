# Generation Model in Speech

## Data

Logmel_normalized_48Label_Framealigned.mat


|train, dev, test| trainlen, devlen, testlen|train_dict,dev_dict,test_dict|
| :-------------: |:-------------:| :-----:|
| log filterbank features of the train, dev and test splits respectively | the length of each utterance in train, dev and test splits respectively |  per-frame aligned labels  |
| (1124823, 122487, 57919)| (3696, 400, 192) | the k-th frame of 'train', its label is the k-th entry of 'train_dict'. The labels are numbers 0-48, where number i corresponds to the (i+1)-th phone in phone-48.txt. Check phone-48.txt to see the 48phones+sil. sil means silence.
|


