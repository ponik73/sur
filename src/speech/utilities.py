from glob import glob
import numpy as np
from numpy import pi, newaxis
from numpy.linalg import norm, inv
from numpy.random import rand
from scipy.io import wavfile
from scipy.special import logsumexp
import scipy.fftpack
import librosa

# Gaussian distributions related functions

def logpdf_gauss(x, mu, cov):
    assert(mu.ndim == 1 and len(mu) == len(cov) and (cov.ndim == 1 or cov.shape[0] == cov.shape[1]))
    x = np.atleast_2d(x) - mu
    if cov.ndim == 1:
        return -0.5*(len(mu)*np.log(2 * pi) + np.sum(np.log(cov)) + np.sum((x**2)/cov, axis=1))
    else:
        return -0.5*(len(mu)*np.log(2 * pi) + np.linalg.slogdet(cov)[1] + np.sum(x.dot(inv(cov)) * x, axis=1))
    
def train_gauss(x):
    """
    Estimates gaussian distribution from data.
    (MU, COV) = TRAIN_GAUSS(X) return Maximum Likelihood estimates of mean
    vector and covariance matrix estimated using training data X
    """
    return np.mean(x, axis=0), np.cov(x.T, bias=True)

# GMM distributions related functions

def logpdf_gmm(x, ws, mus, covs):
    return logsumexp([np.log(w) + logpdf_gauss(x, m, c) for w, m, c in zip(ws, mus, covs)], axis=0)

def train_gmm(x, ws, mus, covs):
    """
    TRAIN_GMM Single iteration of EM algorithm for training Gaussian Mixture Model
    [Ws_new, MUs_new, COVs_new, TLL] = TRAIN_GMM(X, Ws, MUs, COVs) performs single
    iteration of EM algorithm (Maximum Likelihood estimation of GMM parameters)
    using training data X and current model parameters Ws, MUs, COVs and returns 
    updated model parameters Ws_new, MUs_new, COVs_new and total log likelihood
    TLL evaluated using the current (old) model parameters. The model
    parameters are mixture component mean vectors given by columns of M-by-D 
    matrix MUs, covariance matrices given by M-by-D-by-D matrix COVs and vector
    of weights Ws.
    """
    gamma = np.vstack([np.log(w) + logpdf_gauss(x, m, c) for w, m, c in zip(ws, mus, covs)])
    logevidence = logsumexp(gamma, axis=0)
    gamma = np.exp(gamma - logevidence)
    tll = logevidence.sum()
    gammasum = gamma.sum(axis=1)
    ws = gammasum / len(x)
    mus = gamma.dot(x)/gammasum[:,np.newaxis]

    if covs[0].ndim == 1: # diagonal covariance matrices
        covs = gamma.dot(x**2)/gammasum[:,np.newaxis] - mus**2
    else:
        covs = np.array([(gamma[i]*x.T).dot(x)/gammasum[i] - mus[i][:, newaxis].dot(mus[[i]]) for i in range(len(ws))])
    return ws, mus, covs, tll

# Speech features extraction (spectogram, mfcc, ...)

def mel_inv(x):
    return (np.exp(x/1127.)-1.)*700

def mel(x):
    return 1127.*np.log(1. + x / 700.)

def mel_filter_bank(nfft, nbands, fs, fstart=0, fend=None):
    """
    Returns mel filterbank as an array (nfft/2+1 x nbands)
    nfft    - number of samples for FFT computation
    nbands  - number of filter bank bands
    fs      - sampling frequency [Hz]
    fstart  - frequency (Hz) where the first letter starts
    fend    - frequency (Hz) where the last filter ends (default fs/2)
    """
    if not fend:
        fend = 0.5 * fs

    cbin = np.round(mel_inv(np.linspace(mel(fstart), mel(fend), nbands + 2)) / fs * nfft).astype(int)
    mfb = np.zeros((int(nfft / 2 + 1), nbands))
    for ii in range(nbands):
        mfb[cbin[ii]:  cbin[ii+1]+1, ii] = np.linspace(0., 1., cbin[ii+1] - cbin[ii]   +1)
        mfb[cbin[ii+1]:cbin[ii+2]+1, ii] = np.linspace(1., 0., cbin[ii+2] - cbin[ii+1] +1)
    return mfb

def framing(a, window, shift=1):
    shape = (int((a.shape[0] - window) / shift) + 1, window) + a.shape[1:]
    strides = (a.strides[0]*shift, a.strides[0]) + a.strides[1:]
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def spectogram(x, window, noverlap=None, nfft=None):
    if np.isscalar(window): window = np.hamming(window)
    if noverlap is None: noverlap = window.size // 2
    if nfft     is None: nfft     = window.size
    x = framing(x, window.size, window.size-noverlap)
    x = scipy.fftpack.fft(x*window, nfft)
    return x[:,:x.shape[1]//2+1]

def mfcc(s, window, noverlap, nfft, fs, nbanks, nceps):
    """
    MFCC Mel Frequency Cepstral Coefficients
        CPS = MFCC(s, FFTL, Fs, WINDOW, NOVERLAP, NBANKS, NCEPS) returns
        NCEPS-by-M matrix of MFCC coefficients extracted from signal s, where
            M is the number of extracted frames, which can be computed as
            floor((length(S)-NOVERLAP)/(WINDOW_NOVERLAP)). Remaining parameters
            have the following meaning:

        NFFT    - number of frequency points used to calculate the discrete
                  Fourier transforms
        Fs      - sampling frequency [Hz]
        WINDOW  - window length for frame(in samples)
        NOVERLAP    - overlapping between frames (in samples)
        NBANKS  - number of mel filter bank bands
        NCEPS   - number of cepstral coefficients - the output dimensionality
    """

    # Add low level noise (40dB SNR) to avoid log of zeros
    snrdb = 40
    noise = rand(s.shape[0])
    s = s + noise.dot(norm(s, 2)) / norm(noise, 2) / (10 ** (snrdb / 20))

    mfb = mel_filter_bank(nfft, nbanks, fs, 32)
    dct_mx = scipy.fftpack.idct(np.eye(nceps, nbanks), norm='ortho') # the same DCT as in matlab

    S = spectogram(s, window, noverlap, nfft)
    return dct_mx.dot(np.log(mfb.T.dot(np.abs(S.T)))).T

def augment_audio(wav_data, augmentation_factor):
    """
    Augment audio data by time stretching
    """
    wav_data_float = librosa.util.buf_to_float(wav_data)
    augmented_data = librosa.effects.time_stretch(wav_data_float, rate=augmentation_factor)
    return augmented_data

def augment_add_random_noise(wav_data, noise_level=0.01):
    """
    Add random noise to audio data.

    Parameters:
    - wav_data: NumPy array representing the audio signal.
    - noise_level: Magnitude of the random noise (default is 0.01)

    Returns:
    - Noisy audio signal as a NumPy array.
    """
    # Generate random noise with the same length as the audio signal
    noise = noise_level * np.random.rand(len(wav_data))

    # Add the noise to the audio signal
    noisy_audio = wav_data + noise

    return noisy_audio

def wav16khz2mfcc(dir_name, augment=False):
    """
    Loads all *.wav files from directory dir_name (must be 16KHz), converts them into MFCC
    features (13 coefficients) and stores them into a directory. Keys are the file names
    and values and 2D numpy arrays of MFCC features.
    """
    features = {}
    for f in glob(dir_name + '/*.wav'):
        print('Processing file: ', f)
        rate, s = wavfile.read(f)

        # Trim silence
        s_trimmed, _ = librosa.effects.trim(s, top_db=10)
        assert(rate == 16000)
        features[f] = mfcc(s_trimmed, 400, 240, 512, 16000, 23, 13)
        if augment:
            # Data augmentation
            stretch_speeds = [0.5, 0.8, 1.2, 1.5, 2.0]
            for index, speed in enumerate(stretch_speeds):
                augmented_data = augment_audio(s, speed)
                augmented_features = mfcc(augmented_data, 400, 240, 512, 16000, 23, 13)
                augmented_key = f.replace('.wav', '_augmented_stretch_speed_' + str(index) + '.wav')
                features[augmented_key] = augmented_features
            augmented_noisy_data = augment_add_random_noise(s)
            augmented_noisy_features = mfcc(augmented_noisy_data, 400, 240, 512, 16000, 23, 13)
            augmented_noisy_key = f.replace('.wav', '_augmented_random_noise.wav')
            features[augmented_noisy_key] = augmented_noisy_features
    return list(features.keys()), list(features.values())
