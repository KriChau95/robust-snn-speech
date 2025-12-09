import librosa.display
import matplotlib.pyplot as plt

def plot_signal_and_augmented_signal(signal, augmented_signal, sr):
    fig, ax = plt.subplots(nrows=2)
    librosa.display.waveshow(signal, sr=sr, ax=ax[0])
    ax[0].set(title="Original signal")
    librosa.display.waveshow(augmented_signal, sr=sr, ax=ax[1])
    ax[1].set(title="Augmented signal")
    plt.show()

import random
import librosa
import numpy as np
import soundfile as sf

def add_white_noise(signal, noise_percentage_factor):
    noise = np.random.normal(0, signal.std(), signal.size)
    augmented_signal = signal + noise * noise_percentage_factor
    return augmented_signal

def time_stretch(signal, time_stretch_rate):
    return librosa.effects.time_stretch(signal, time_stretch_rate)

def pitch_scale(signal, sr, num_semitones):
    return librosa.effects.pitch_shift(signal, sr, num_semitones)

def random_gain(signal, min_factor=0.1, max_factor=0.12):
    gain_rate = random.uniform(min_factor, max_factor)
    augmented_signal = signal * gain_rate
    return augmented_signal

def invert_polarity(signal):
    return signal * -1

if __name__ == "__main__":
    signal, sr = librosa.load("speech_commands/cat/0ab3b47d_nohash_0.wav")
    augmented_signal = add_white_noise(signal, 0.5)
    sf.write("augmented_audio.wav", augmented_signal, sr)
    plot_signal_and_augmented_signal(signal, augmented_signal, sr)