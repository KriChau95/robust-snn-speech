import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

# Define model parameters

num_inputs = 20
num_hidden = 256
num_classes = 2
num_steps = 200

dtype = torch.float
batch_size = 25
num_epochs = 40
learning_rate = 5e-4

high_rate = 200 / 1000
low_rate  = 20 / 1000

save_dir = "processed_spike_data"

class ANN(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(num_inputs * 2, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_hidden)
        self.fc3 = nn.Linear(num_hidden, num_hidden)
        self.fc4 = nn.Linear(num_hidden, num_classes)

        self.relu = nn.ReLU()

    def forward(self, x):
        x_rate = x.mean(dim=-1)               
        x_rate = x_rate.view(x.size(0), -1)  

        x = self.relu(self.fc1(x_rate))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        out = self.fc4(x)

        return out                         


def load_spike_data(filename):
    return np.load(os.path.join(save_dir, filename), allow_pickle=True)

if __name__ == "__main__":

    X_train_pos = load_spike_data("X_train_pos.npy")
    X_train_neg = load_spike_data("X_train_neg.npy")
    y_train = np.load(os.path.join(save_dir, "y_train.npy"))

    X_test_pos = load_spike_data("X_test_pos.npy")
    X_test_neg = load_spike_data("X_test_neg.npy")
    y_test = np.load(os.path.join(save_dir, "y_test.npy"))

    print(f"Data Loaded: Train = {len(X_train_pos)}, Test = {len(X_test_pos)}")

    label_map = {"cat": 0, "dog": 1}

    train_dataset = []
    test_dataset = []

    # Build train dataset
    for i in range(len(X_train_pos)):
        blank = torch.zeros((2, num_inputs, num_steps), dtype=dtype)

        for neuron, t in X_train_pos[i]:
            blank[0, neuron, t - 1] = 1.0
        for neuron, t in X_train_neg[i]:
            blank[1, neuron, t - 1] = 1.0

        label = y_train[i]
        if label in label_map:
            train_dataset.append((blank, label_map[label]))


    # Build test dataset
    for i in range(len(X_test_pos)):
        blank = torch.zeros((2, num_inputs, num_steps), dtype=dtype)

        for neuron, t in X_test_pos[i]:
            blank[0, neuron, t - 1] = 1.0
        for neuron, t in X_test_neg[i]:
            blank[1, neuron, t - 1] = 1.0

        label = y_test[i]
        if label in label_map:
            test_dataset.append((blank, label_map[label]))

    print(f"Filtered: Train = {len(train_dataset)}, Test = {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, drop_last=True)

    # Define model parameters
    net = ANN()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    train_loss_hist, test_loss_hist = [], []
    train_acc_hist, test_acc_hist = [], []

    # Training Loop
    for epoch in tqdm(range(num_epochs)):

        net.train()
        epoch_loss = 0
        correct = 0
        total = 0

        for data, targets in train_loader:

            target_rate = torch.full(
                (data.size(0), num_classes), low_rate, dtype=dtype
            )

            target_rate[range(data.size(0)), targets] = high_rate

            outputs = net(data)
            loss = criterion(outputs, target_rate)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            _, preds = outputs.max(1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

        train_loss = epoch_loss / len(train_loader)
        train_acc = correct / total

        train_loss_hist.append(train_loss)
        train_acc_hist.append(train_acc)

        
        # Evaluation during training

        net.eval()
        test_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, targets in test_loader:

                target_rate = torch.full(
                    (data.size(0), num_classes), low_rate, dtype=dtype
                )
                target_rate[range(data.size(0)), targets] = high_rate

                outputs = net(data)
                loss = criterion(outputs, target_rate)

                test_loss += loss.item()

                _, preds = outputs.max(1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)

        test_loss /= len(test_loader)
        test_acc = correct / total

        test_loss_hist.append(test_loss)
        test_acc_hist.append(test_acc)

        # Print epoch results

        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss {train_loss:.6f} | Test Loss {test_loss:.6f} | "
            f"Train Acc {train_acc*100:.2f}% | Test Acc {test_acc*100:.2f}%"
        )

    # Save the trained model
    torch.save(net.state_dict(), "ann.pth")

    # Plot training and test loss and accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss_hist, label="Train Loss")
    plt.plot(test_loss_hist, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(train_acc_hist, label="Train Accuracy")
    plt.plot(test_acc_hist, label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()