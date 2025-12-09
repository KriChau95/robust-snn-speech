import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import snntorch as snn
from snntorch import surrogate, functional as SF
from tqdm import tqdm

# =========================
# CONFIG
# =========================

save_dir = "processed_spike_data"   # where your .npy files are
batch_size = 25
dtype = torch.float

num_inputs = 80     # neurons (rows)
num_steps = 200     # time steps (columns)

num_hidden = 256
beta = 0.95         # LIF decay
num_epochs = 20
learning_rate = 5e-4

# Binary classification
label_map = {
    "cat": 0,
    "yes": 1,
}
num_classes = 2
valid_labels = set(label_map.keys())

# =========================
# DATA LOADING
# =========================

def load_spike_data(filename):
    return np.load(os.path.join(save_dir, filename), allow_pickle=True)

def load_labels(filename):
    return np.load(os.path.join(save_dir, filename), allow_pickle=True)

print("Loading spike data from:", save_dir)
X_train_pos_loaded = load_spike_data("X_train_pos.npy")
X_train_neg_loaded = load_spike_data("X_train_neg.npy")
y_train_loaded = load_labels("y_train.npy")

X_test_pos_loaded = load_spike_data("X_test_pos.npy")
X_test_neg_loaded = load_spike_data("X_test_neg.npy")
y_test_loaded = load_labels("y_test.npy")

print(f"Data Loaded: Train Samples = {len(X_train_pos_loaded)}, Test Samples = {len(X_test_pos_loaded)}")

# =========================
# BUILD DATASETS
# Each sample -> tensor [80, 200], label int {0,1}
# =========================

train_dataset, test_dataset = [], []

train_size = len(X_train_pos_loaded)
test_size = len(X_test_pos_loaded)

# ---- build train dataset ----
for i in range(train_size):
    blank_tensor = torch.zeros((num_inputs, num_steps), dtype=dtype)

    pos_spikes = X_train_pos_loaded[i]   # array of [num_spikes, 2] (neuron, timestep)
    neg_spikes = X_train_neg_loaded[i]
    label = y_train_loaded[i]

    if label in valid_labels:
        # positive spikes
        for neuron, timestep in pos_spikes:
            n = int(neuron)
            t = int(timestep) - 1   # timesteps 1..200 -> 0..199
            if 0 <= n < num_inputs and 0 <= t < num_steps:
                blank_tensor[n, t] = 1.0

        # negative spikes (keep as -1; you can experiment with removing them later)
        for neuron, timestep in neg_spikes:
            n = int(neuron)
            t = int(timestep) - 1
            if 0 <= n < num_inputs and 0 <= t < num_steps:
                blank_tensor[n, t] = -1.0

        train_dataset.append((blank_tensor, label_map[label]))

# ---- build test dataset ----
for i in range(test_size):
    blank_tensor = torch.zeros((num_inputs, num_steps), dtype=dtype)

    pos_spikes = X_test_pos_loaded[i]
    neg_spikes = X_test_neg_loaded[i]
    label = y_test_loaded[i]

    if label in valid_labels:
        for neuron, timestep in pos_spikes:
            n = int(neuron)
            t = int(timestep) - 1
            if 0 <= n < num_inputs and 0 <= t < num_steps:
                blank_tensor[n, t] = 1.0

        for neuron, timestep in neg_spikes:
            n = int(neuron)
            t = int(timestep) - 1
            if 0 <= n < num_inputs and 0 <= t < num_steps:
                blank_tensor[n, t] = -1.0

        test_dataset.append((blank_tensor, label_map[label]))

print(f"Filtered Data: Train Samples = {len(train_dataset)}, Test Samples = {len(test_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

# quick sanity check
for data, target in train_loader:
    print(f"Example batch shapes -> data: {data.size()}, target: {target.size()}")
    break

# =========================
# MODEL DEFINITION
# =========================

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
        """
        x: [batch, 80, 200]
        returns: spk4_rec [T, B, num_classes]
        """
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()

        spk4_rec = []

        for step in range(num_steps):
            x_t = x[:, :, step]    # [B, 80]

            cur1 = self.fc1(x_t)
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)

            cur4 = self.fc4(spk3)
            spk4, mem4 = self.lif4(cur4, mem4)

            spk4_rec.append(spk4)

        # shape: [T, B, num_classes]
        return torch.stack(spk4_rec, dim=0)

# =========================
# TRAINING SETUP
# =========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

net = Net().to(device)

# Classification loss on spike counts (big fix vs MSE!)
criterion = SF.ce_count_loss()

optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, betas=(0.9, 0.999))

# =========================
# TRAINING LOOP
# =========================

for epoch in range(num_epochs):
    net.train()
    total_loss = 0.0
    total_acc = 0.0
    total_samples = 0

    print(f"\nEpoch {epoch+1}/{num_epochs}")

    for data, targets in tqdm(train_loader, desc="Train", leave=False):
        data = data.to(device)
        targets = targets.to(device)

        spk_rec = net(data)                  # [T, B, 2]
        loss_val = criterion(spk_rec, targets)
        acc = SF.accuracy_rate(spk_rec, targets)

        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        batch_size_now = data.size(0)
        total_loss += loss_val.item() * batch_size_now
        total_acc += acc.item() * batch_size_now
        total_samples += batch_size_now

    train_loss = total_loss / total_samples
    train_acc = total_acc / total_samples

    # ---- EVAL ON TEST SET ----
    net.eval()
    test_loss = 0.0
    test_acc = 0.0
    test_samples = 0

    with torch.no_grad():
        for data, targets in tqdm(test_loader, desc="Test", leave=False):
            data = data.to(device)
            targets = targets.to(device)

            spk_rec = net(data)
            loss_val = criterion(spk_rec, targets)
            acc = SF.accuracy_rate(spk_rec, targets)

            bsz = data.size(0)
            test_loss += loss_val.item() * bsz
            test_acc += acc.item() * bsz
            test_samples += bsz

    test_loss /= test_samples
    test_acc /= test_samples

    print(
        f"Epoch {epoch} | "
        f"Train Loss: {train_loss:.6f} | Train Acc: {train_acc*100:.2f}% | "
        f"Test Loss: {test_loss:.6f} | Test Acc: {test_acc*100:.2f}%"
    )
