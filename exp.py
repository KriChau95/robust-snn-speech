# train_snn_fixed.py
"""
Fixed SNN training script (Two-channel spike encoding).
- Channel 0: positive spikes (1)
- Channel 1: negative spikes (1)
- No -1 values fed into the network.
- Precompute fc1 input once per sample (static input across time)
- Loss averaged over time steps
- Use membrane potentials summed over time as logits for classification
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import snntorch as snn

# ---------------------------
# Config / Hyperparameters
# ---------------------------
save_dir = "processed_spike_data"   # adjust if needed
batch_size = 10
dtype = torch.float32
num_epochs = 10
lr = 5e-4
beta = 0.95
num_steps = 200   # temporal dimension
num_neurons = 80
num_timesteps = 200  # should match your encoding
num_channels = 2  # positive + negative channels

label_map = {
    "cat": 0,
    "dog": 1,
    "go": 2,
    "no": 3,
    "yes": 4,
}

num_outputs = len(label_map)
num_inputs = num_channels * num_neurons * num_timesteps
num_hidden = 1000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ---------------------------
# Helpers: load spike arrays
# ---------------------------
def load_spike_data(filename):
    return np.load(os.path.join(save_dir, filename), allow_pickle=True)

# load numpy spike lists
X_train_pos_loaded = load_spike_data("X_train_pos.npy")
X_train_neg_loaded = load_spike_data("X_train_neg.npy")
y_train_loaded = np.load(os.path.join(save_dir, "y_train.npy"), allow_pickle=True)

X_test_pos_loaded = load_spike_data("X_test_pos.npy")
X_test_neg_loaded = load_spike_data("X_test_neg.npy")
y_test_loaded = np.load(os.path.join(save_dir, "y_test.npy"), allow_pickle=True)

train_size = len(X_train_pos_loaded)
test_size = len(X_test_pos_loaded)
print(f"Data Loaded: Train samples = {train_size}, Test samples = {test_size}")

# ---------------------------
# Build datasets with 2-channel encoding
# shape: (channels, neurons, timesteps) -> (2, 80, 200)
# ---------------------------
def build_dataset(X_pos, X_neg, y_array):
    dataset = []
    for i in range(len(X_pos)):
        # two channels: pos and neg
        tensor = torch.zeros((num_channels, num_neurons, num_timesteps), dtype=torch.float32)

        pos_spikes = X_pos[i]
        neg_spikes = X_neg[i]

        # pos_spikes and neg_spikes are assumed lists/arrays of (neuron, timestep)
        for neuron, timestep in pos_spikes:
            # timestep in file is 1..T -> convert to 0-based
            t = int(timestep) - 1
            n = int(neuron)
            tensor[0, n, t] = 1.0

        for neuron, timestep in neg_spikes:
            t = int(timestep) - 1
            n = int(neuron)
            tensor[1, n, t] = 1.0

        label_str = y_array[i]
        label = label_map[label_str]
        # store as (channels, neurons, timesteps) -> model will flatten
        dataset.append((tensor, label))
    return dataset

train_dataset = build_dataset(X_train_pos_loaded, X_train_neg_loaded, y_train_loaded)
test_dataset = build_dataset(X_test_pos_loaded, X_test_neg_loaded, y_test_loaded)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

# ---------------------------
# Network
# ---------------------------
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=beta)

    def forward(self, x):
        # x shape: (B, channels, neurons, timesteps)
        B = x.size(0)
        x_flat = x.view(B, -1)  # (B, num_inputs)

        # precompute first-layer current for static input
        cur1_base = self.fc1(x_flat)  # (B, num_hidden)

        # initialize membrane states (use defaults from snntorch)
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        spk2_rec = []
        mem2_rec = []

        for step in range(num_steps):
            # feed same (static) cur1_base into lif1 across steps (if input is static)
            spk1, mem1 = self.lif1(cur1_base, mem1)
            cur2 = self.fc2(spk1)            # (B, num_outputs)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        # returns tensors shaped (T, B, C)
        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)


# Instantiate model, loss, optimizer
net = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)

# ---------------------------
# Utility functions
# ---------------------------
def compute_accuracy_from_mem(mem_rec, targets):
    """
    mem_rec: (T, B, C)
    targets: (B,)
    """
    # sum membrane potentials across time -> (B, C)
    logits = mem_rec.sum(dim=0)
    preds = logits.argmax(dim=1)
    correct = (preds == targets).sum().item()
    return correct / targets.size(0)

# ---------------------------
# Training loop
# ---------------------------
loss_hist = []
test_loss_hist = []

for epoch in range(num_epochs):
    net.train()
    running_loss = 0.0
    running_acc = 0.0
    n_batches = 0

    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device)                     # (B, 2, 80, 200)
        targets = torch.tensor(targets, dtype=torch.long, device=device)

        optimizer.zero_grad()
        spk_rec, mem_rec = net(data)               # spk_rec, mem_rec: (T, B, C)

        # compute loss: average over time steps
        losses = []
        for step in range(num_steps):
            losses.append(criterion(mem_rec[step], targets))
        loss_val = torch.stack(losses).mean()

        loss_val.backward()
        # optional gradient clipping to stabilize training
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)
        optimizer.step()

        # stats
        batch_acc = compute_accuracy_from_mem(mem_rec, targets)
        running_loss += loss_val.item()
        running_acc += batch_acc
        n_batches += 1
        loss_hist.append(loss_val.item())

        # optionally print per-batch
        if batch_idx % 20 == 0:
            print(f"[Epoch {epoch+1}/{num_epochs}] Batch {batch_idx}, Loss {loss_val.item():.4f}, Batch Acc {batch_acc*100:.2f}%")

    train_loss = running_loss / max(1, n_batches)
    train_acc = running_acc / max(1, n_batches)
    print(f"Epoch {epoch+1} TRAIN loss: {train_loss:.4f}, TRAIN acc: {train_acc*100:.2f}%")

    # ---------------------------
    # Evaluate on test set
    # ---------------------------
    net.eval()
    test_running_loss = 0.0
    test_running_acc = 0.0
    test_batches = 0

    with torch.no_grad():
        for data, targets in test_loader:
            data = data.to(device)
            targets = torch.tensor(targets, dtype=torch.long, device=device)

            spk_rec_t, mem_rec_t = net(data)

            # average loss over time
            losses_t = [criterion(mem_rec_t[step], targets) for step in range(num_steps)]
            loss_t = torch.stack(losses_t).mean()
            test_running_loss += loss_t.item()

            batch_acc = compute_accuracy_from_mem(mem_rec_t, targets)
            test_running_acc += batch_acc
            test_batches += 1
            test_loss_hist.append(loss_t.item())

    if test_batches > 0:
        test_loss = test_running_loss / test_batches
        test_acc = test_running_acc / test_batches
    else:
        test_loss = 0.0
        test_acc = 0.0

    print(f"Epoch {epoch+1}  TEST  loss: {test_loss:.4f}, TEST acc: {test_acc*100:.2f}%")
    print("-" * 60)

# ---------------------------
# Save model (optional)
# ---------------------------
model_path = "snn_model_two_channel.pth"
torch.save(net.state_dict(), model_path)
print("Model saved to", model_path)
