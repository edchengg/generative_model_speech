
import scipy.io
import numpy as np
import gc


def w_frame(w, train, train_len):
    train_data = []
    start = 0
    tmp = []
    # for each utterance
    for idx in range(len(train_len)):
        len_num = train_len[idx]
        end = start + int(len_num)
        utterance = train[start:end]
        size = len(utterance[0])
        length = len(utterance)
        sub_train_data = []
        i = 0
        # check each frame in the utterance
        while i <= length-1:
            # if silence, pass
            if utterance[i][-1] == 37:
                i += 1
            # select frames with same label
            else:
                # use pointer to find same label array
                label = utterance[i][-1]
                j = i
                while utterance[j][-1] == label and j < length - 1:
                    j += 1
                # find mid idx
                mid = i + (j - i) // 2
                # get left part and right part and concatenate together
                # padding
                if mid - w//2 >= 0:
                    w_frame_left = utterance[(mid-w//2):mid]
                else:
                    extra = abs((mid - w//2)) + 1
                    silence = np.zeros((extra,size))
                    silence[:,-1]=37
                    w_frame_left = utterance[i:mid]
                    w_frame_left = np.concatenate((silence, w_frame_left),axis=0)

                if mid + w//2 + 1 <= length:
                    w_frame_right = utterance[mid:(mid+w//2)+1]
                else:
                    extra = (mid + w//2)+1 - length
                    tmp.append(extra)
                    silence = np.zeros((extra,size))
                    silence[:,-1]=37
                    w_frame_right = utterance[mid:]
                    w_frame_right = np.concatenate((w_frame_right, silence),axis=0)

                w_frame = np.concatenate((w_frame_left,w_frame_right), axis=0)

                if len(w_frame) != w:
                    print('*****ERRROR******')
                sub_train_data.append(w_frame)
                # check and update i
                if i != j:
                    i = j
                else:
                    i += 1
                # force garbage collection
                del w_frame_right, w_frame_left, w_frame
                gc.collect()
        # append into data list and force garbage collection
        train_data.append(sub_train_data)
        del sub_train_data
        del utterance
        gc.collect()
        start = end
        print('complete: %f'%(idx/len(train_len)*100))

    return np.asanyarray(train_data)

def transfer(add):
    ## transfer from N * X * 9 * 121 TO  Y * 1080 without label
    data_train = np.load(add + '.npy')
    data_p = []
    for i in range(len(data_train)):
        for j in range(len(data_train[i])):
            data_p.append(np.asarray(data_train[i][j][:,:-1]).flatten())
    np.save(add+'_nolabel',np.asarray(data_p))


if __name__ == "__main__":
    #mat = scipy.io.loadmat('/share/data/speech/Data/timit_matlab/Logmel_normalized_48Label_Framealigned.mat')
    mat = scipy.io.loadmat('Logmel_normalized_48Label_Framealigned.mat')
    print('processing.....')
    #train_data = mat['train']
    #train_len = mat['trainlen']
    #train_dict = mat['train_dict']
    #train = np.concatenate((train_data, train_dict), axis=1)
    #del train_data, train_dict

    #dev_data = mat['dev']
    #dev_len = mat['devlen']
    #dev_dict = mat['dev_dict']
    #dev = np.concatenate((dev_data, dev_dict), axis=1)
    #del dev_data, dev_dict

    test_data = mat['test']
    test_len = mat['testlen']
    test_dict = mat['test_dict']
    test = np.concatenate((test_data,test_dict),axis=1)
    del test_data, test_dict

    #train_data_list = w_frame(9, train, train_len)
    #np.save('train_data', train_data_list)
    #del train_data_list
    #print('train data done')
    #dev_data_list = w_frame(9, dev, dev_len)
    #np.save('dev_data', dev_data_list)
    #del dev_data_list
    #print('dev data done')
    test_data_list = w_frame(9, test, test_len)
    np.save('test_data', test_data_list)
    del test_data_list
    print('test data done')

    #transfer('train_data')

    #transfer('dev_data')

    transfer('test_data')
