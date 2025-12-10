import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np

import os

from snntorch import functional as SF
from snntorch import surrogate
from tqdm import tqdm

num_inputs = 20
num_hidden = 256
num_classes = 2

num_steps = 200
beta = 0.95
spike_grad = surrogate.fast_sigmoid()
dtype = torch.float

save_dir = 'processed_spike_data'
wn_save_dir = 'pitched_processed_spike_data'

class Net(nn.Module):

    # Define network layers (4 fully connected layers with LIF neurons)
    def __init__(self):

        super().__init__()

        self.fc1 = nn.Linear(num_inputs*2, num_hidden)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        self.fc2 = nn.Linear(num_hidden, num_hidden)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        self.fc3 = nn.Linear(num_hidden, num_hidden)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        self.fc4 = nn.Linear(num_hidden, num_classes)
        self.lif4 = snn.Leaky(beta=beta, spike_grad=spike_grad)
    
    # Run the network over all time steps and record output spikes
    def forward(self, x):

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()

        spk4_rec = []

        # Iterate over time dimension and update spiking layers
        for step in range(num_steps):

            x_t = x[:, :, :, step].reshape(x.size(0), -1)

            cur1 = self.fc1(x_t)
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)

            cur4 = self.fc4(spk3)
            spk4, mem4 = self.lif4(cur4, mem4)

            spk4_rec.append(spk4)

        return torch.stack(spk4_rec, dim=0)

# Load spike data from the original (clean) dataset
def load_spike_data(filename):
    loaded_container = np.load(os.path.join(save_dir, filename), allow_pickle=True)
    return loaded_container

# Load spike data from the pitched (augmented) dataset
def load_wn_spike_data(filename):
    loaded_container = np.load(os.path.join(wn_save_dir, filename), allow_pickle=True)
    return loaded_container

if __name__== "__main__":

    X_train_pos_loaded = load_spike_data('X_train_pos.npy')
    X_train_neg_loaded = load_spike_data('X_train_neg.npy')
    y_train_loaded = np.load(os.path.join(save_dir, 'y_train.npy'))

    X_test_pos_loaded = load_spike_data('X_test_pos.npy')
    X_test_neg_loaded = load_spike_data('X_test_neg.npy')
    y_test_loaded = np.load(os.path.join(save_dir, 'y_test.npy'))

    wn_X_train_pos_loaded = load_wn_spike_data('X_train_pos.npy')
    wn_X_train_neg_loaded = load_wn_spike_data('X_train_neg.npy')
    wn_y_train_loaded = np.load(os.path.join(wn_save_dir, 'y_train.npy'))

    wn_X_test_pos_loaded = load_wn_spike_data('X_test_pos.npy')
    wn_X_test_neg_loaded = load_wn_spike_data('X_test_neg.npy')
    wn_y_test_loaded = np.load(os.path.join(wn_save_dir, 'y_test.npy'))

    print(f"Data Loaded: Train Samples = {len(X_train_pos_loaded)}, Test Samples = {len(X_test_pos_loaded)}")
    batch_size = 25

    train_size = len(X_train_pos_loaded)
    test_size = len(X_test_pos_loaded)

    wn_train_size = len(wn_X_train_pos_loaded)
    wn_test_size = len(wn_X_test_pos_loaded)

    train_dataset, test_dataset = [], []
    wn_train_dataset, wn_test_dataset = [], []

    label_map = {
        'cat' : 0,
        'dog' : 1,
    }

    # Build spike tensors and labels for each sample in the original training set
    for i in range(train_size):

        blank_tensor = torch.zeros((2, num_inputs, num_steps), dtype=dtype)  # 2 channels: 0=pos, 1=neg
        
        pos_spikes = X_train_pos_loaded[i]
        neg_spikes = X_train_neg_loaded[i]
        label = y_train_loaded[i]

        if label in list(label_map.keys()):

            # Fill positive/negative channel with spikes from pos_spikes/neg_spikes list
            for neuron, timestep in pos_spikes:
                blank_tensor[0, neuron, timestep-1] = 1.0 
            for neuron, timestep in neg_spikes:
                blank_tensor[1, neuron, timestep-1] = 1.0
            
            train_dataset.append((blank_tensor, label_map[label]))
    
    # Build spike tensors and labels for each sample in the original test set
    for i in range(test_size):

        blank_tensor = torch.zeros((2, num_inputs, num_steps), dtype=dtype)  # 2 channels: 0=pos, 1=neg
        
        pos_spikes = X_test_pos_loaded[i]
        neg_spikes = X_test_neg_loaded[i]
        label = y_test_loaded[i]

        if label in list(label_map.keys()):
            
            # Fill positive/negative channel with spikes from pos_spikes/neg_spikes list
            for neuron, timestep in pos_spikes:
                blank_tensor[0, neuron, timestep-1] = 1.0
            for neuron, timestep in neg_spikes:
                blank_tensor[1, neuron, timestep-1] = 1.0
            
            test_dataset.append((blank_tensor, label_map[label]))

    # Build spike tensors and labels for each sample in the pitched training set
    for i in range(wn_train_size):
        
        blank_tensor = torch.zeros((2, num_inputs, num_steps), dtype=dtype)  # 2 channels: 0=pos, 1=neg
        
        pos_spikes = wn_X_train_pos_loaded[i]
        neg_spikes = wn_X_train_neg_loaded[i]
        label = wn_y_train_loaded[i]

        if label in list(label_map.keys()):
            
            # Fill positive/negative channel with spikes from pos_spikes/neg_spikes list
            for neuron, timestep in pos_spikes:
                blank_tensor[0, neuron, timestep-1] = 1.0
            for neuron, timestep in neg_spikes:
                blank_tensor[1, neuron, timestep-1] = 1.0
            
            wn_train_dataset.append((blank_tensor, label_map[label]))
    
    # Build spike tensors and labels for each sample in the pitched test set
    for i in range(wn_test_size):
        
        blank_tensor = torch.zeros((2, num_inputs, num_steps), dtype=dtype)  # 2 channels: 0=pos, 1=neg
        
        pos_spikes = wn_X_test_pos_loaded[i]
        neg_spikes = wn_X_test_neg_loaded[i]
        label = wn_y_test_loaded[i]

        if label in list(label_map.keys()):
            
            # Fill positive/negative channel with spikes from pos_spikes/neg_spikes list
            for neuron, timestep in pos_spikes:
                blank_tensor[0, neuron, timestep-1] = 1.0
            for neuron, timestep in neg_spikes:
                blank_tensor[1, neuron, timestep-1] = 1.0
            
            wn_test_dataset.append((blank_tensor, label_map[label]))

    full_train_dataset = train_dataset + wn_train_dataset
    full_test_dataset = test_dataset + wn_test_dataset

    train_loader = DataLoader(full_train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(full_test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    print(f"Filtered Data: Train Samples = {len(full_train_dataset)}, Test Samples = {len(full_test_dataset)}")

    # Loop through one batch to check tensor shapes
    for data, target in train_loader:
        print(f"Data batch shape: {data.size()}")
        print(f"Target batch shape: {target.size()}")
        break

    net = Net()

    high_rate = 200/1000
    low_rate = 20/1000

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=5e-4, betas=(0.9, 0.999))

    num_epochs = 40
    train_loss_hist = []
    test_loss_hist = []
    train_acc_hist = []
    test_acc_hist = []
    counter = 0

    # Train and evaluate the network for multiple epochs
    for epoch in tqdm(range(num_epochs)):

        net.train()
        epoch_loss = 0
        correct_train = 0
        total_train  = 0

        # Training loop over all batches
        for data, targets in tqdm(train_loader):

            local_target_rate = torch.full((data.size(0), num_classes), low_rate, dtype=dtype)
            local_target_rate[range(data.size(0)), targets] = high_rate

            spk_rec = net(data)
            actual_rate = torch.sum(spk_rec, dim=0) / num_steps

            loss_val = criterion(actual_rate, local_target_rate)
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            epoch_loss += loss_val.item()

            total_spikes = spk_rec.sum(dim=0)
            _, preds = total_spikes.max(1)
            correct_train += (preds == targets).sum().item()
            total_train += data.size(0)
        
        train_loss = epoch_loss / len(train_loader)
        train_loss_hist.append(train_loss)
        train_acc = correct_train / total_train
        train_acc_hist.append(train_acc)

        with torch.no_grad():
            net.eval()
            test_loss = 0
            correct = 0
            total_samples = 0

            # Evaluation loop over all test batches
            for data, targets in test_loader:

                test_local_target_rate = torch.full((data.size(0), num_classes), low_rate, dtype=dtype)
                test_local_target_rate[range(data.size(0)), targets] = high_rate

                test_spk_rec = net(data)
                actual_rate = torch.sum(test_spk_rec, dim=0) / num_steps

                loss_val = criterion(actual_rate, test_local_target_rate)

                test_loss += loss_val.item()

                total_spikes = test_spk_rec.sum(dim=0)
                _, preds = total_spikes.max(1)
                correct += (preds == targets).sum().item()
                total_samples += data.size(0)
        
            test_loss /= len(test_loader)
            test_loss_hist.append(test_loss)
            test_acc = correct / total_samples
            test_acc_hist.append(test_acc)

        print(f"Epoch {epoch} | Train Loss: {train_loss:.6f} | Test Loss: {test_loss_hist[-1]:.6f} | Train Acc: {train_acc*100:.2f}% | Test Acc: {test_acc*100:.2f}%")

    torch.save(net.state_dict(), "pitch_snn.pth")

    plt.figure(figsize=(10,5))
    plt.plot(train_loss_hist, label="Train Loss")
    plt.plot(test_loss_hist, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train vs Test Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10,5))
    plt.plot(train_acc_hist, label="Train Accuracy")
    plt.plot(test_acc_hist, label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Train vs Test Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()