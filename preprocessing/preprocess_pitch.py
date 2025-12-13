import os
import random
import numpy as np
import torchaudio
from speech2spikes import S2S
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import Counter

# Initialize parameters for data, splitting, and loading

script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.abspath(os.path.join(script_dir, "..", "pitched_speech_commands"))

train_ratio = 0.8
data_per_class = 500

# function to get all .wav files needed for training and testing
# returns list of train files, test files, train labels, test labels
def get_train_test_files_with_labels(data_path, train_ratio=0.8, data_per_class=100):

    # Initalize empty lists
    train_files, train_labels = [], []
    test_files, test_labels = [], []

    classes = os.listdir(data_path)
    
    # for each class folder, get files and split into train/test
    for c in classes:

        cls_path = os.path.join(data_path, c)

        files = [os.path.join(cls_path, f) for f in os.listdir(cls_path)][:data_per_class]

        random.shuffle(files)
        
        split_idx = int(len(files) * train_ratio)
        
        train_files += files[:split_idx]
        train_labels += [c] * len(files[:split_idx])
        
        test_files += files[split_idx:]
        test_labels += [c] * len(files[split_idx:])
    
    return (train_files, train_labels), (test_files, test_labels)

# Get train and test files with labels
(train_files, train_labels), (test_files, test_labels) = get_train_test_files_with_labels(data_path, train_ratio, data_per_class)

print(f"Train: {len(train_files)} files, Test: {len(test_files)} files")
print("Sample train files and labels:", list(zip(train_files[:5], train_labels[:5])))

# function that uses Speech2Spikes to convert .wav files to spike trains
def spikes_from_files_with_labels(files, labels, s2s):

    pos_list, neg_list, file_labels = [], [], []

    for file, label in tqdm(zip(files, labels), total=len(files)):
        waveform, _ = torchaudio.load(file)
        spike_train = s2s([(waveform, 'None')])[0]
        
        spike_pos = (spike_train == 1)
        spike_neg = (spike_train == -1)
        
        raw_pos = np.argwhere(spike_pos.squeeze().numpy() == 1)
        raw_neg = np.argwhere(spike_neg.squeeze().numpy() == 1)

        if raw_pos.size > 0 or raw_neg.size > 0:
            max_t = 0
            if raw_pos.size > 0:
                max_t = max(max_t, raw_pos[:,1].max())
            if raw_neg.size > 0:
                max_t = max(max_t, raw_neg[:,1].max())

            if max_t > 0:
                raw_pos[:,1] = (raw_pos[:,1] * 199 / max_t).astype(int)
                raw_neg[:,1] = (raw_neg[:,1] * 199 / max_t).astype(int)

        pos_list.append(raw_pos)
        neg_list.append(raw_neg)

        file_labels.append(label)

    return pos_list, neg_list, file_labels

# Initialize Speech2Spikes object
s2s = S2S(labels=['None'])

# Convert train and test sets into spikes
X_train_pos, X_train_neg, y_train = spikes_from_files_with_labels(train_files, train_labels, s2s)
X_test_pos, X_test_neg, y_test = spikes_from_files_with_labels(test_files, test_labels, s2s)

# function to help visualize spike trains
def plot_spike_train(pos_spikes, neg_spikes, file_path, label):

    plt.figure(figsize=(12,3))
    plt.scatter(pos_spikes[:,1], pos_spikes[:,0], color='red', s=1, label='Pos Spike')
    plt.scatter(neg_spikes[:,1], neg_spikes[:,0], color='blue', s=1, label='Neg Spike')
    
    plt.title(f"Spike Train for {label}: {os.path.basename(file_path)}")
    plt.xlabel("Time")
    plt.ylabel("Neuron")
    
    plt.legend()
    plt.show()

# Save the processed spike data and labels

# Define the directory for saving data
save_dir = "./pitched_processed_spike_data"
os.makedirs(save_dir, exist_ok=True)

# function to save spike data
def save_spike_data(data_list, filename):
    data_to_save = np.array(data_list, dtype=object) 
    np.save(os.path.join(save_dir, filename), data_to_save, allow_pickle=True)

# function to save label data
def save_labels(data_list, filename):
    np.save(os.path.join(save_dir, filename), np.array(data_list))

# Save training data

save_spike_data(X_train_pos, 'X_train_pos.npy')
save_spike_data(X_train_neg, 'X_train_neg.npy')

save_labels(y_train, 'y_train.npy')

# Save testing data

save_spike_data(X_test_pos, 'X_test_pos.npy')
save_spike_data(X_test_neg, 'X_test_neg.npy')

save_labels(y_test, 'y_test.npy')

print("All training and testing data successfully saved.")

print(len(X_train_pos))

display_sample_plot = False

if display_sample_plot: plot_spike_train(X_train_pos[0], X_train_neg[0], train_files[0], y_train[0])
