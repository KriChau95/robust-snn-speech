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
    
    train_dataset.append((blank_tensor, label_map[y_train_loaded[i]]))

for i in range(test_size):

    blank_tensor = torch.zeros((80, 200), dtype=dtype)
    
    pos_spikes = X_test_pos_loaded[i]
    neg_spikes = X_test_neg_loaded[i]

    for neuron, timestep in pos_spikes:
        blank_tensor[neuron, timestep-1] = 1.0 # minus one because timesteps range from from 1 - 200
    
    for neuron, timestep in neg_spikes:
        blank_tensor[neuron, timestep-1] = -1.0
    
    test_dataset.append((blank_tensor, label_map[y_test_loaded[i]]))

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

num_inputs = 80
num_hidden = 256

num_steps = 200
beta = 0.95

spike_grad = surrogate.fast_sigmoid()

class Net(nn.Module):

    def __init__(self):

        super().__init__()

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

        for step in range(num_steps):

            x_t = x[:, :, step]
            cur1 = self.fc1(x_t)
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)

            cur4 = self.fc4(spk3)
            spk4, mem4 = self.lif4(cur4, mem4)

            spk4_rec.append(spk4)

        return torch.stack(spk4_rec, dim=0)  # output shape: (T, B, N_classes)

net = Net()

high_rate = 200/1000
low_rate = 20/1000

criterion = nn.CrossEntropyLoss()  # expects spike_counts: [B, C], targets: [B]
optimizer = torch.optim.Adam(net.parameters(), lr=5e-4, betas=(0.9, 0.999))

def calculate_accuracy(spk_rec, targets):

    total_spikes = spk_rec.sum(dim = 0)

    _, idx = total_spikes.max(1)

    acc = np.mean((targets == idx).detach().cpu().numpy())
    return acc

def print_batch_accuracy(spk_rec, targets, train = False):
    acc = calculate_accuracy(spk_rec, targets)
    print(f"{'Train' if train else 'Test'} set accuracy for a single minibatch: {acc*100:.2f}%")

num_epochs = 20
loss_hist = []
test_loss_hist = []
counter = 0

for epoch in tqdm(range(num_epochs)):

    net.train()
    epoch_loss = 0
    correct_train = 0
    total_train  = 0

    for data, targets in tqdm(train_loader):

        spk_rec = net(data)
        spike_counts = spk_rec.sum(dim=0)

        loss_val = criterion(spike_counts, targets)
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        epoch_loss += loss_val.item()

        _, pred = spike_counts.max(1)
        correct_train += (pred == targets).sum().item()
        total_train += data.size(0)
    
    train_loss = epoch_loss / len(train_loader)
    train_acc = correct_train / total_train

    with torch.no_grad():
        net.eval()
        test_loss = 0
        correct_test = 0
        total_test = 0
        for data, targets in test_loader:

            spk_rec = net(data)
            spike_counts = spk_rec.sum(dim=0)

            loss_val = criterion(spike_counts, targets)

            test_loss += loss_val.item()

            _, pred = spike_counts.max(1)
            correct_test += (pred == targets).sum().item()
            total_test += data.size(0)
        
        test_loss_hist.append(test_loss / len(test_loader))
        test_acc = correct_test / total_test

    print(f"Epoch {epoch} | Train Loss: {train_loss:.6f} | Test Loss: {test_loss_hist[-1]:.6f} | Train Acc: {train_acc*100:.2f}% | Test Acc: {test_acc*100:.2f}%")