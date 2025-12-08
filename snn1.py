import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np
import itertools

import os

from snntorch import functional as SF
from snntorch import surrogate
from tqdm import tqdm

save_dir = 'processed_spike_data'

def load_spike_data(filename):
    loaded_container = np.load(os.path.join(save_dir, filename), allow_pickle=True)
    return loaded_container

X_train_pos_loaded = load_spike_data('X_train_pos.npy')
X_train_neg_loaded = load_spike_data('X_train_neg.npy')
y_train_loaded = np.load(os.path.join(save_dir, 'y_train.npy'))

X_test_pos_loaded = load_spike_data('X_test_pos.npy')
X_test_neg_loaded = load_spike_data('X_test_neg.npy')
y_test_loaded = np.load(os.path.join(save_dir, 'y_test.npy'))

print(f"Data Loaded: Train Samples = {len(X_train_pos_loaded)}, Test Samples = {len(X_test_pos_loaded)}")

batch_size = 25
dtype = torch.float

train_size = len(X_train_pos_loaded)
test_size = len(X_test_pos_loaded)

num_classes = 5

train_dataset, test_dataset = [], []

label_map = {
    'cat' : 0,
    'dog' : 1,
    'go' : 2,
    'no' : 3,
    'yes' : 4,
}

for i in range(train_size):

    blank_tensor = torch.zeros((80, 200), dtype=dtype)
    
    pos_spikes = X_train_pos_loaded[i]
    neg_spikes = X_train_neg_loaded[i]

    for neuron, timestep in pos_spikes:
        blank_tensor[neuron, timestep-1] = 1.0 # minus one because timesteps range from from 1 - 200
    
    for neuron, timestep in neg_spikes:
        blank_tensor[neuron, timestep-1] = -1.0
    
    train_dataset.append((blank_tensor.unsqueeze(0), label_map[y_train_loaded[i]]))

for i in range(test_size):

    blank_tensor = torch.zeros((80, 200), dtype=dtype)
    
    pos_spikes = X_test_pos_loaded[i]
    neg_spikes = X_test_neg_loaded[i]

    for neuron, timestep in pos_spikes:
        blank_tensor[neuron, timestep-1] = 1.0 # minus one because timesteps range from from 1 - 200
    
    for neuron, timestep in neg_spikes:
        blank_tensor[neuron, timestep-1] = -1.0
    
    test_dataset.append((blank_tensor.unsqueeze(0), label_map[y_test_loaded[i]]))

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

num_inputs = 80 * 200
num_hidden = 256

num_steps = 200
beta = 0.95

spike_grad = surrogate.fast_sigmoid()

class Net(nn.Module):

    def __init__(self):

        super().__init__()

        self.dropout = nn.Dropout(p=0.5)

        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        self.fc2 = nn.Linear(num_hidden, num_hidden)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        self.fc3 = nn.Linear(num_hidden, num_hidden)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        self.fc4 = nn.Linear(num_hidden, num_classes)
        self.lif4 = snn.Leaky(beta=beta, spike_grad=spike_grad)
    
    def forward(self, x):

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()

        spk4_rec = []

        x_flat = x.view(batch_size, -1) # reshape to Batch size x (80*200)

        for step in range(num_steps):

            cur1 = self.fc1(x_flat)
            spk1, mem1 = self.lif1(cur1, mem1)
            spk1_drop = self.dropout(spk1) # Apply dropout after spiking

            cur2 = self.fc2(spk1_drop)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_drop = self.dropout(spk2) # Apply dropout after spiking

            cur3 = self.fc3(spk2_drop)
            spk3, mem3 = self.lif3(cur3, mem3)

            cur4 = self.fc4(spk3)
            spk4, mem4 = self.lif4(cur4, mem4)

            spk4_rec.append(spk4)

        return torch.stack(spk4_rec, dim=0)  # output shape: (T, B, N_classes)

net = Net()

high_rate = 50/1000
low_rate = 5/1000

target_rate = torch.full((batch_size, num_classes), low_rate, dtype = dtype)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=5e-4, betas=(0.9, 0.999), weight_decay=1e-4)

def calculate_accuracy(spk_rec, targets):

    total_spikes = spk_rec.sum(dim = 0)

    _, idx = total_spikes.max(1)

    acc = np.mean((targets == idx).detach().cpu().numpy())
    return acc

def print_batch_accuracy(spk_rec, targets, train = False):
    acc = calculate_accuracy(spk_rec, targets)
    print(f"{'Train' if train else 'Test'} set accuracy for a single minibatch: {acc*100:.2f}%")

def train_printer():
    print(f"Epoch {epoch}, Iteration {iter_counter}")
    print(f"Train Set Loss: {loss_hist[counter]:.7f}")
    print(f"Test Set Loss: {test_loss_hist[counter]:.7f}")
    print_batch_accuracy(spk_rec, targets, train=True)
    print_batch_accuracy(test_spk, test_targets, train=False)
    print("\n")

num_epochs = 10
loss_hist = []
test_loss_hist = []
counter = 0

for epoch in tqdm(range(num_epochs)):

    iter_counter = 0
    train_batch = iter(train_loader)

    for data, targets in train_batch:

        local_target_rate = target_rate.clone()

        local_target_rate[range(batch_size), targets] = high_rate

        net.train()
        spk_rec = net(data)

        actual_rate = torch.sum(spk_rec, dim=0) / num_steps

        loss_val = criterion(actual_rate, local_target_rate)

        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        loss_hist.append(loss_val.item())

        with torch.no_grad():

            net.eval()

            test_data, test_targets = next(iter(test_loader))

            test_local_target_rate = torch.full((batch_size, num_classes), low_rate, dtype=dtype)
            test_local_target_rate[range(batch_size), test_targets] = high_rate

            test_spk = net(test_data)

            test_actual_rate = torch.sum(test_spk, dim=0) / num_steps
            test_loss = criterion(test_actual_rate, test_local_target_rate)

            test_loss_hist.append(test_loss.item())

            if counter % 10 == 0:
                train_printer()
            counter += 1
            iter_counter +=1












