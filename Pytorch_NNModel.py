import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
from sklearn.dummy import DummyClassifier
from tqdm import tqdm
import os


class NNModel(nn.Module):
    def __init__(self, in_dim, n_classes):
        super(NNModel, self).__init__()
        self.fc1 = nn.Linear(in_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, n_classes)
        self.activation = nn.LeakyReLU(0.01)

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = torch.softmax(self.fc4(x), dim=1)
        return x

    def train_model(
        self, train_loader, criterion, optimizer, epochs, checkpoint_dir=None
    ):
        if checkpoint_dir is not None and not os.path.exists(checkpoint_dir):
            print("creating directory")
            os.makedirs(checkpoint_dir)
        for epoch in range(epochs):
            running_loss = 0.0
            correct_predictions = 0
            total_samples = 0
            progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
            for i, (inputs, labels) in progress_bar:
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(
                    outputs, labels.clone().detach().long()
                )  # torch.tensor(labels, dtype=torch.long))
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                # Calculate accuracy
                _, predicted = torch.max(outputs, 1)
                train_acc = torch.sum(predicted == labels)
                # correct_predictions += (predicted == labels).sum().item()
                accuracy = 100 * train_acc / outputs.size(0)
                progress_bar.set_description(
                    f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / (i+1):.4f}, Acc: {accuracy:.2f}%"
                )
            # Save checkpoint
            if checkpoint_dir is not None:
                checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch+1}.pt")
                torch.save(
                    self.state_dict(),
                    checkpoint_path,
                )
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader):.4f}")

    def predict(self, data_loader):
        predictions = []
        with torch.no_grad():
            progress_bar = tqdm(enumerate(data_loader), total=len(data_loader))
            for i, (inputs, labels) in progress_bar:
                outputs = self(inputs)
                _, predicted = torch.max(outputs, 1)
                predictions.extend(predicted.tolist())
        return predictions

    def print_num_parameters(self):
        num_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Number of parameters in the model: {num_parameters}")


def prepare_data(train_data, labels):
    tensor_x = torch.tensor(train_data, dtype=torch.float32)
    tensor_y = torch.tensor(labels)
    dataset = TensorDataset(tensor_x, tensor_y)
    return dataset


# Dummy model for comparison
class DummyModel:
    def __init__(self):
        self.dummy_clf = DummyClassifier(strategy="stratified", random_state=2987)

    def train(self, train_data, y):
        self.dummy_clf.fit(train_data, y)

    def predict(self, pred_data):
        return self.dummy_clf.predict(pred_data)
