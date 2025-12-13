from ann import ANN
import torch
from torch.utils.data import DataLoader
import os
import numpy as np

net = ANN() 
net.load_state_dict(torch.load("ann.pth"))
net.eval()

num_inputs = 20
num_classes = 2
high_rate = 200/1000
low_rate = 20/1000
num_steps = 200
dtype = torch.float

# Load spike data arrays given a file name
def load_spike_data(filename):
    loaded_container = np.load(os.path.join(save_dir, filename), allow_pickle=True)
    return loaded_container

save_dir = 'wn_processed_spike_data'

X_train_pos_loaded = load_spike_data('X_train_pos.npy')
X_train_neg_loaded = load_spike_data('X_train_neg.npy')
y_train_loaded = np.load(os.path.join(save_dir, 'y_train.npy'))

X_test_pos_loaded = load_spike_data('X_test_pos.npy')
X_test_neg_loaded = load_spike_data('X_test_neg.npy')
y_test_loaded = np.load(os.path.join(save_dir, 'y_test.npy'))

print(f"Data Loaded: Train Samples = {len(X_train_pos_loaded)}, Test Samples = {len(X_test_pos_loaded)}")
batch_size = 25

train_size = len(X_train_pos_loaded)
test_size = len(X_test_pos_loaded)

dataset = []

label_map = {
    'cat' : 0,
    'dog' : 1,
}

# Build spike tensors and labels for each sample in the training set
for i in range(train_size):

    blank_tensor = torch.zeros((2, num_inputs, num_steps), dtype=dtype)  # 2 channels: 0=pos, 1=neg
    
    pos_spikes = X_train_pos_loaded[i]
    neg_spikes = X_train_neg_loaded[i]
    label = y_train_loaded[i]

    if label in list(label_map.keys()):

        for neuron, timestep in pos_spikes:
            blank_tensor[0, neuron, timestep-1] = 1.0  # pos channel
        for neuron, timestep in neg_spikes:
            blank_tensor[1, neuron, timestep-1] = 1.0  # neg channel
        
        dataset.append((blank_tensor, label_map[label]))

# Build spike tensors and labels for each sample in the test set
for i in range(test_size):

    blank_tensor = torch.zeros((2, num_inputs, num_steps), dtype=dtype)  # 2 channels: 0=pos, 1=neg
    
    pos_spikes = X_test_pos_loaded[i]
    neg_spikes = X_test_neg_loaded[i]
    label = y_test_loaded[i]

    if label in list(label_map.keys()):

        for neuron, timestep in pos_spikes:
            blank_tensor[0, neuron, timestep-1] = 1.0  # pos channel
        for neuron, timestep in neg_spikes:
            blank_tensor[1, neuron, timestep-1] = 1.0  # neg channel
        
        dataset.append((blank_tensor, label_map[label]))

original_data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

print(f"Samples = {len(dataset)}")

with torch.no_grad():
    net.eval()
    correct = 0
    total_samples = 0

    for data, targets in original_data_loader:

        outputs = net(data)          # ANN outputs rates directly
        _, preds = outputs.max(1)

        correct += (preds == targets).sum().item()
        total_samples += targets.size(0)

    test_acc = correct / total_samples
    print("Accuracy on white noise data: {:.2f}%".format(test_acc * 100))

