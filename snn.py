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

save_dir = 'processed_spike_data'

def load_spike_data(filename):
    loaded_container = np.load(os.path.join(save_dir, filename), allow_pickle=True)
    return loaded_container 

# Load Training Data
X_train_pos_loaded = load_spike_data('X_train_pos.npy')
X_train_neg_loaded = load_spike_data('X_train_neg.npy')
y_train_loaded = np.load(os.path.join(save_dir, 'y_train.npy'))

# Load Testing Data
X_test_pos_loaded = load_spike_data('X_test_pos.npy')
X_test_neg_loaded = load_spike_data('X_test_neg.npy')
y_test_loaded = np.load(os.path.join(save_dir, 'y_test.npy'))

print(f"Data Loaded: Train Samples = {len(X_train_pos_loaded)}, Test Samples = {len(X_test_pos_loaded)}")

batch_size = 10
dtype = torch.float

train_size = len(X_train_pos_loaded)
test_size = len(X_test_pos_loaded)

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

""" REMOVE LATER: Note on data format for SNNs
If you plan to use a simple Fully Connected (Dense) SNN, you might need to reshape this:Time-first format (RNN-style): 
Reshape to $(T, B, N)$, i.e., $(200, B, 80)$. This is done by permuting the tensor before 
feeding it into the model.Flattened format: Flatten the last two dimensions to $(B, 80 \t
imes 200)$."""

# Network Architecture
num_inputs = 80*200
num_hidden = 256
num_outputs = 5

# Temporal Dynamics
num_steps = 200
beta = 0.95

# Define Network
class Net(nn.Module):
    def __init__(self):
        super().__init__()

        # Initialize layers
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=beta)

    def forward(self, x):

        # Initialize hidden states at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        # Record the final layer
        spk2_rec = []
        mem2_rec = []

        for step in range(num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)

net = Net()

def print_batch_accuracy(data, targets, train=False):
    output, _ = net(data.view(batch_size, -1))
    _, idx = output.sum(dim=0).max(1)
    acc = np.mean((targets == idx).detach().cpu().numpy())

    if train:
        print(f"Train set accuracy for a single minibatch: {acc*100:.2f}%")
    else:
        print(f"Test set accuracy for a single minibatch: {acc*100:.2f}%")

def train_printer():
    print(f"Epoch {epoch}, Iteration {iter_counter}")
    print(f"Train Set Loss: {loss_hist[counter]:.2f}")
    print(f"Test Set Loss: {test_loss_hist[counter]:.2f}")
    print_batch_accuracy(data, targets, train=True)
    print_batch_accuracy(test_data, test_targets, train=False)
    print("\n")

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=5e-4, betas=(0.9, 0.999))

num_epochs = 10
loss_hist = []
test_loss_hist = []
counter = 0

# Outer training loop
for epoch in range(num_epochs):
    iter_counter = 0
    train_batch = iter(train_loader)

    # Minibatch training loop
    for data, targets in train_batch:

        # forward pass
        net.train()
        spk_rec, mem_rec = net(data.view(batch_size, -1))

        # initialize the loss & sum over time
        loss_val = torch.zeros((1), dtype=dtype)
        for step in range(num_steps):
            loss_val += loss(mem_rec[step], targets)

        # Gradient calculation + weight update
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        # Store loss history for future plotting
        loss_hist.append(loss_val.item())

        # Test set
        with torch.no_grad():
            net.eval()
            test_data, test_targets = next(iter(test_loader))
            test_data = test_data
            test_targets = test_targets

            # Test set forward pass
            test_spk, test_mem = net(test_data.view(batch_size, -1))

            # Test set loss
            test_loss = torch.zeros((1), dtype=dtype)
            for step in range(num_steps):
                test_loss += loss(test_mem[step], test_targets)
            test_loss_hist.append(test_loss.item())

            # Print train/test loss/accuracy
            if counter % 5 == 0:
                train_printer()
            counter += 1
            iter_counter +=1