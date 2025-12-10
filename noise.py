import librosa.display
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import numpy as np
import random
import librosa
import numpy as np
import soundfile as sf

# Plot original and augmented waveforms one above the other
def plot_signal_and_augmented_signal(signal, augmented_signal, sr):
    fig, ax = plt.subplots(nrows=2)
    librosa.display.waveshow(signal, sr=sr, ax=ax[0])
    ax[0].set(title="Original signal")
    librosa.display.waveshow(augmented_signal, sr=sr, ax=ax[1])
    ax[1].set(title="Augmented signal")
    plt.subplots_adjust(hspace=0.4)
    plt.show()

# Add Gaussian white noise scaled by noise_percentage_factor
def add_white_noise(signal, noise_percentage_factor):
    noise = np.random.normal(0, signal.std(), signal.size)
    augmented_signal = signal + noise * noise_percentage_factor
    return augmented_signal

# Stretch/compress the audio in time without changing pitch
def time_stretch(signal, time_stretch_rate):
    return librosa.effects.time_stretch(signal, time_stretch_rate)

# Shift the pitch of the audio by num_semitones
def pitch_scale(signal, sr, num_semitones):
    return librosa.effects.pitch_shift(y=signal, sr=sr, n_steps=num_semitones)

# Randomly scale the volume of the signal between min_factor and max_factor
def random_gain(signal, min_factor=0.1, max_factor=0.12):
    gain_rate = random.uniform(min_factor, max_factor)
    augmented_signal = signal * gain_rate
    return augmented_signal

# Flip the signal vertically (invert its amplitude)
def invert_polarity(signal):
    return signal * -1

signal, sr = librosa.load("speech_commands/cat/0ab3b47d_nohash_0.wav")
augmented_signal = pitch_scale(signal, sr, -20)
sf.write("augmented_audio.wav", augmented_signal, sr)
plot_signal_and_augmented_signal(signal, augmented_signal, sr)

input_dir = "speech_commands"
output_dir = "white_noise_speech_commands"

os.makedirs(output_dir, exist_ok=True)

samples_per_word = 500

# Loop over each word subfolder and create white-noise-augmented versions
for word in os.listdir(input_dir):

    word_path = os.path.join(input_dir, word)
    distorted_word_path = os.path.join(output_dir, word)

    os.makedirs(distorted_word_path, exist_ok=True)

    audio_files = os.listdir(word_path)

    indices = np.random.choice(len(audio_files), size=samples_per_word, replace=False)

    # For each selected file, load it, add noise, and save the result
    for i in indices:
        base_file = os.path.join(word_path, audio_files[i])
        signal, sr = librosa.load(base_file)

        distorted_signal = add_white_noise(signal, noise_percentage_factor=0.5)
        output_file = os.path.join(distorted_word_path, f"wn_{i}_{os.path.basename(base_file)}")
        sf.write(output_file, distorted_signal, sr)

input_dir = "speech_commands"
output_dir = "pitched_speech_commands"

os.makedirs(output_dir, exist_ok=True)

samples_per_word = 500

# Loop over each word subfolder and create pitch-shifted versions
for word in os.listdir(input_dir):

    word_path = os.path.join(input_dir, word)
    distorted_word_path = os.path.join(output_dir, word)

    os.makedirs(distorted_word_path, exist_ok=True)

    audio_files = os.listdir(word_path)

    indices = np.random.choice(len(audio_files), size=samples_per_word, replace=False)

    # For each selected file, load it, apply random pitch shift, and save the result
    for i in indices:
        base_file = os.path.join(word_path, audio_files[i])
        signal, sr = librosa.load(base_file)

        distorted_signal = pitch_scale(signal, sr, random.randint(-5, 5))
        output_file = os.path.join(distorted_word_path, f"wn_{i}_{os.path.basename(base_file)}")
        sf.write(output_file, distorted_signal, sr)