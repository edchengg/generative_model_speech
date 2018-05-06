import matplotlib.pyplot as plt
PLOT_CONFIG = { 'interpolation': "nearest", 'aspect': "auto", 'cmap': "Greys" }


import numpy as np

from numpy.fft import ifft
from scipy.fftpack import dct, idct

from collections import defaultdict
from copy import deepcopy
from glob import glob


def pre_emphasis(x):
    """
    Applies pre-emphasis step to the signal.
    - balance frequencies in spectrum by increasing amplitude of high frequency
    bands and decreasing the amplitudes of lower bands
    - largely unnecessary in modern feature extraction pipelines
    ------
    :in:
    x, array of samples
    ------
    :out:
    y, array of samples
    """
    y = np.append(x[0], x[1:] - 0.97 * x[:-1])

    return y


def hamming(n):
    """
    Hamming method for weighting components of window.
    Feel free to implement more window functions.
    ------
    :in:
    n, window size
    ------
    :out:
    win, array of weights to apply along window
    """
    win = 0.54 - .46 * np.cos(2 * np.pi * np.arange(n) / (n - 1))

    return win


def windowing(x, size, step):
    """
    Window and stack signal into overlapping frames.
    ------
    :in:
    x, array of samples
    size, window size in number of samples (Note: this may need to be a power of 2)
    step, window shift in number of samples
    ------
    :out:
    frames, 2d-array of frames with shape (number of windows, window size)
    """
    xpad = np.append(x, np.zeros((size - len(x) % size)))

    T = (len(xpad) - size) // step
    frames = np.stack([xpad[t * step:t * step + size] for t in range(T)])

    return frames


def discrete_fourier_transform(x):
    """
    Compute the discrete fourier transform for each frame of windowed signal x.
    Typically, we talk about performing the DFT on short-time windows
    (often referred to as the Short-Time Fourier Transform). Here, the input
    is a 2d-array with shape (window size,  number of windows). We want to
    perform the DFT on each of these windows.
    Note: this can be done in a vectorized form or in a loop.
    --------
    :in:
    x, 2d-array of frames with shape (window size, number of windows)
    --------
    :out:
    X, 2d-array of complex spectrum after DFT applied to each window of x
    """

    # TODO: Implement DFT
    fft_size = x.shape[0]
    j = np.arange(fft_size)
    i = j.reshape((fft_size, 1))
    M = np.exp(-2j * np.pi * i * j / fft_size)
    X = np.dot(M, x)
    return X


def fast_fourier_transform(x):
    """
    Fast-fourier transform. Effiicient algorithm for computing the DFT.
    --------
    :in:
    x, 2d-array of frames with shape (window size, number of windows)
    --------
    :out:
    X, 2d-array of complex spectrum after DFT applied to each window of x
    """
    fft_size = len(x)

    if fft_size <= 16:
        X = discrete_fourier_transform(x)

    else:
        indices = np.arange(fft_size)
        even = fast_fourier_transform(x[::2])
        odd = fast_fourier_transform(x[1::2])
        m = np.exp(-2j * np.pi * indices / fft_size).reshape(-1, 1)
        X = np.concatenate([even + m[:fft_size // 2] * odd, even + m[fft_size // 2:] * odd])

    return X


def mel_filterbank(nfilters, fft_size, sample_rate):
    """
    Mel-warping filterbank.
    You do not need to edit this code; it is needed to contruct the mel filterbank
    which we will use to extract features.
    --------
    :in:
    nfilters, number of filters
    fft_size, window size over which fft is performed
    sample_rate, sampling rate of signal
    --------
    :out:
    mel_filter, 2d-array of (fft_size / 2, nfilters) used to get mel features
    mel_inv_filter, 2d-array of (nfilters, fft_size / 2) used to invert
    melpoints, 1d-array of frequencies converted to mel-scale
    """
    freq2mel = lambda f: 2595. * np.log10(1 + f / 700.)
    mel2freq = lambda m: 700. * (10 ** (m / 2595.) - 1)

    lowfreq = 0
    highfreq = sample_rate // 2

    lowmel = freq2mel(lowfreq)
    highmel = freq2mel(highfreq)

    melpoints = np.linspace(lowmel, highmel, 1 + nfilters + 1)

    # must convert from freq to fft bin number
    fft_bins = ((fft_size + 1) * mel2freq(melpoints) // sample_rate).astype(np.int32)

    filterbank = np.zeros((nfilters, fft_size // 2))
    for j in range(nfilters):
        for i in range(fft_bins[j], fft_bins[j + 1]):
            filterbank[j, i] = (i - fft_bins[j]) / (fft_bins[j + 1] - fft_bins[j])
        for i in range(fft_bins[j + 1], fft_bins[j + 2]):
            filterbank[j, i] = (fft_bins[j + 2] - i) / (fft_bins[j + 2] - fft_bins[j + 1])

    mel_filter = filterbank.T / filterbank.sum(axis=1).clip(1e-16)
    mel_inv_filter = filterbank

    return mel_filter, mel_inv_filter, melpoints


def inv_spectrogram(X_s, size, step, n_iter=15):
    """
    Feel free to disregard this code. It is not necessary that
    you follow the code below, but it can be used to invert
    from the spectrogram (signal spectrum magnitude) back to the signal
    which can be helpful when qualitatively assessing the nature of
    compression into MFCC features.
    """

    def find_offset(a, b):
        corrs = np.convolve(a - a.mean(), b[::-1] - b.mean())
        corrs[:len(b) // 2] = -1e12
        corrs[-len(b) // 2:] = -1e12
        return corrs.argmax() - len(a)

    def iterate(X, iteration):
        T, n = X.shape
        size = n // 2

        x = np.zeros((T * step + size))
        window_sum = np.zeros((T * step + size))

        est_start = size // 2 - 1
        est_stop = est_start + size

        for t in range(T):
            x_start = t * step
            x_stop = x_start + size

            est = ifft(X[t].real + 0j if iteration == 0 else X[t]).real[::-1]
            if t > 0 and x_stop - step > x_start and est_stop - step > est_start:
                offset = find_offset(x[x_start:x_stop - step], est[est_start:est_stop - step])
            else:
                offset = 0

            x[x_start:x_stop] += est[est_start - offset:est_stop - offset] * hamming(size)
            window_sum[x_start:x_stop] += hamming(size)

        return x.real / window_sum.clip(1e-12)

    X_s = np.concatenate([X_s, X_s[:, ::-1]], axis=1)
    reg = np.max(X_s) / 1e8

    X_best = iterate(deepcopy(X_s), 0)
    for i in range(1, n_iter):
        X_best = windowing(X_best, size, step) * hamming(size)
        est = fast_fourier_transform(X_best.T).T
        phase = est / np.maximum(reg, np.abs(est))
        X_best = iterate(X_s * phase[:len(X_s)], i)

    return np.real(X_best)

def recover_sound(data,iteration, direct):
    size = 256 # window size for the FFT
    step = size // 2 # distance to slide along the window in time
    nfilters = 40 # number of mel frequency channels
    ncoeffs = 13 # number of cepstral coeffecients to keep
    fs = 16000

    data = data.data.numpy()
    mel_filter, mel_inv_filter, melpoints = mel_filterbank(nfilters, size, fs)
    #mfccs = dct(data.reshape((9, 120))[:, :40], type=2, axis=1, norm='ortho')
    #mfccs = dct(data.reshape((9,40)), type=2, axis=1, norm='ortho')

    # invert from MFCCs back to waveform
    #recovered_log_mel_fbank = idct(mfccs, type=2, n=nfilters, axis=1, norm='ortho')

    # exponentiate log and invert mel warping
    recovered_power = (10**data.reshape(19,40)).dot(mel_inv_filter)

    # invert mel warping of spectrogram
    recovered_magnitude = np.sqrt(recovered_power * size)

    recovered_signal = inv_spectrogram(recovered_magnitude, size, step)

    # Look at specgram
    fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(1,4))
    cax = ax.matshow(20*np.log10(recovered_magnitude.clip(1e-12)).T, origin='lower', **PLOT_CONFIG)
    fig.colorbar(cax, label='dB')
    ax.grid(False)
    plt.title('log spectrogram MelFilter Bank (dB)')
    plt.xlabel('# Frames')
    plt.ylabel('Indices')
    NQF = fs / 2
    indices = np.arange(0, recovered_magnitude.shape[1], 8)
    frequencies = np.arange(0, NQF, NQF * 8 / recovered_magnitude.shape[1])
    plt.ylabel('Hz')
    plt.yticks(indices, frequencies)
    plt.savefig('./figures/'+direct+'/spect_'+str(iteration))
    plt.close()