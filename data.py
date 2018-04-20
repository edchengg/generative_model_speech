
import scipy.io
import numpy as np
import gc
# 1892966

def w_frame(w, train, train_len):
    train_data = []
    start = 0
    for idx in range(len(train_len)):
        len_num = train_len[idx]
        end = start + int(len_num)
        utterance = train[start:end]
        size = len(utterance[0])
        length = len(utterance)
        sub_train_data = []
        i = 0
        while i <= length-1:
            if utterance[i][-1] == 37:
                i += 1
            else:
                # use pointer to find same label array
                label = utterance[i][-1]
                j = i
                while utterance[j][-1] == label and j < length - 1:
                    j += 1
                mid = i + (j - i) // 2

                # padding
                try:
                    w_frame_left = utterance[(mid-w//2):mid]
                except IndexError:
                    extra = abs((mid - w//2)) + 1
                    silence = np.zeros((extra,size))
                    silence[:,-1]=37
                    w_frame_left = utterance[i:mid]
                    w_frame_left = np.concatenate((silence, w_frame_left),axis=0)

                try:
                    w_frame_right = utterance[mid:(mid+w//2)+1]
                except IndexError:
                    extra = (mid + w//2)+1 - length
                    silence = np.zeros((extra,size))
                    silence[:,-1]=37
                    w_frame_right = utterance[mid:]
                    w_frame_right = np.concatenate((w_frame_right, silence),axis=0)

                w_frame = np.concatenate((w_frame_left,w_frame_right), axis=0)
                sub_train_data.append(w_frame)

                if i != j:
                    i = j
                else:
                    i += 1

                del w_frame_right, w_frame_left, w_frame
                gc.collect()

        train_data.append(sub_train_data)
        del sub_train_data
        del utterance
        gc.collect()
        start = end
        print('complete: %f'%(idx/len(train_len)*100))

    return np.asanyarray(train_data)

if __name__ == "__main__":
    #mat = scipy.io.loadmat('/share/data/speech/Data/timit_matlab/Logmel_normalized_48Label_Framealigned.mat')
    mat = scipy.io.loadmat('Logmel_normalized_48Label_Framealigned.mat')
    print('processing.....')
    train_data = mat['train']
    train_len = mat['trainlen']
    train_dict = mat['train_dict']
    train = np.concatenate((train_data, train_dict), axis=1)
    del train_data, train_dict

    dev_data = mat['dev']
    dev_len = mat['devlen']
    dev_dict = mat['dev_dict']
    dev = np.concatenate((dev_data, dev_dict), axis=1)
    del dev_data, dev_dict

    test_data = mat['test']
    test_len = mat['testlen']
    test_dict = mat['test_dict']
    test = np.concatenate((test_data,test_dict),axis=1)
    del test_data, test_dict

    train_data_list = w_frame(9, train, train_len)
    np.save('train_data', train_data_list)
    del train_data_list
    print('train data done')
    dev_data_list = w_frame(9, dev, dev_len)
    np.save('dev_data', dev_data_list)
    del dev_data_list
    print('dev data done')
    test_data_list = w_frame(9, test, test_len)
    np.save('test_data', test_data_list)
    del test_data_list
    print('test data done')
