import scipy
import numpy as np
import random
import librosa

class FourierTransform():
    """
    Contains the fourier transform (stft), the amplitude spectrogran(spec), the phase angles(phase) and the sample rate(srate) of an input signal

    init takes in an audio signal(arr), along with it's sampling rate (srate) and the target fft sample size(n_fft)

    random_shuffle == True randomly shuffles the spectrograms along the time axis to prevent the network from using the information along the time axis
    """
    def __init__(self, array, srate = 22050, n_fft = 256, random_shuffle = False) -> None:

        #https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.stft.html
        #scipy.signal.stft returns array of sample frequencies, array of sample times and the short term fourier transform of a signal
        self.f, self.t, self.stft = scipy.signal.stft(array, fs = srate, nperseg = n_fft)
        
        if random_shuffle==True:
            np.random.default_rng().shuffle(self.stft, axis = 1)
        
        self.spec = np.abs(self.stft)

        self.phase = np.angle(self.stft)

        self.srate = srate


class Audio():
    """
        contains an audio sample(array), it's sample rate(srate) and it's FourierTransform(ft)

        load_from_file reads an audio sample from file and resamples it to a target sample rate (srate). 
        If random_shuffle == True, segments of segment_size are randomly pitch shifted up or down
    """

    def __init__ (self, filename, srate = 22050, n_fft = 256, shuffle_spec = False, shuffle_audio = False, resample_step_size = 1000):
        self.array, self.srate = librosa.load(filename, sr = srate)

        
        if shuffle_audio:
            self.array = random_resample_audio(self.array, self.srate, step_size = resample_step_size)
        
        self.array = np.divide(self.array, np.amax(np.abs(self.array)))

        self.ft = FourierTransform(array = self.array,srate = self.srate, n_fft = n_fft, random_shuffle = shuffle_spec)

        self.n_fft = n_fft

        



def random_resample_audio(array, sample_rate = 22050, step_size = 256):
    iterator = 0

    #the function will append the resampled audio segments to this audio_array and return it
    audio_array = []

    while iterator <= array.shape[0] - step_size:

        #scales the original sample rate by a random factor ranging from 1/2 to 2 times the original
        rand_rate = random.randint(int(sample_rate/2), sample_rate*2)

        #https://librosa.org/doc/main/generated/librosa.resample.html
        #resampling is handled by librosa.resample
        audio_array.extend(librosa.resample(array[iterator:iterator+step_size], orig_sr = sample_rate, target_sr = rand_rate))

        iterator += step_size

    return np.array(audio_array)

