import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import snntorch as snn
from snntorch import surrogate, functional as SF
from tqdm import tqdm

save_dir = "processed_spike_data"
batch_size = 25
dtype = torch.float

num_inputs = 80
num_steps = 200

num_hidden = 256
beta = 0.95
num_epochs = 20
learning_rate = 5e-4

# choose what two words to use
label_map = {
    "no": 0,
    "go": 1,
}

num_classes = 2
valid_labels = set(label_map.keys())


# Load spike data (pos/neg spike arrays) from disk
def load_spike_data(filename):
    return np.load(os.path.join(save_dir, filename), allow_pickle=True)

# Load label arrays from disk
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

train_dataset, test_dataset = [], []

train_size = len(X_train_pos_loaded)
test_size = len(X_test_pos_loaded)

# Convert list-of-spikes format into [80, 200] spike tensors + integer labels
for i in range(train_size):
    blank_tensor = torch.zeros((num_inputs, num_steps), dtype=dtype)

    pos_spikes = X_train_pos_loaded[i]
    neg_spikes = X_train_neg_loaded[i]
    label = y_train_loaded[i]

    if label in valid_labels:

        # positive spikes: set entries to +1 at (neuron, timestep)
        for neuron, timestep in pos_spikes:
            n = int(neuron)
            t = int(timestep) - 1
            if 0 <= n < num_inputs and 0 <= t < num_steps:
                blank_tensor[n, t] = 1.0

        # negative spikes: set entries to -1 at (neuron, timestep)
        for neuron, timestep in neg_spikes:
            n = int(neuron)
            t = int(timestep) - 1
            if 0 <= n < num_inputs and 0 <= t < num_steps:
                blank_tensor[n, t] = -1.0

        train_dataset.append((blank_tensor, label_map[label]))

# Same processing as above but for test data
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

# Quick check on batch shapes
for data, target in train_loader:
    print(f"Example batch shapes -> data: {data.size()}, target: {target.size()}")
    break

spike_grad = surrogate.fast_sigmoid()

class Net(nn.Module):

    # Define a 4-layer fully connected SNN with LIF neurons
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

    # Run the network over all time steps and return output spike trains
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

        # Step through time and update LIF layers at each step
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

        return torch.stack(spk4_rec, dim=0)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

net = Net().to(device)

criterion = SF.ce_count_loss()

optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, betas=(0.9, 0.999))

# Main training + testing loop across epochs
for epoch in range(num_epochs):
    net.train()
    total_loss = 0.0
    total_acc = 0.0
    total_samples = 0

    print(f"\nEpoch {epoch+1}/{num_epochs}")

    # Training loop over all mini-batches
    for data, targets in tqdm(train_loader, desc="Train", leave=False):
        data = data.to(device)
        targets = targets.to(device)

        spk_rec = net(data)
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

    net.eval()
    test_loss = 0.0
    test_acc = 0.0
    test_samples = 0

    # Evaluation loop on the test set
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
