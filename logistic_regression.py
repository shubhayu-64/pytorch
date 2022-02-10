import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.data import random_split
from torchvision.datasets import MNIST


class MNISTModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, xb):
        xb = xb.reshape(-1, 784)
        out = self.linear(xb)
        return out

    def train(self, batch):
        image, label = batch
        out = self(image)
        loss = F.cross_entropy(out, label)
        return loss

    def validate(self, batch):
        image, label = batch
        preds = self(image)
        loss = F.cross_entropy(preds, label)
        acc = accuracy(preds, label)
        return {"loss": loss, "acc": acc}

    def compute_metric(self, preds):
        batch_loss = [batch["loss"] for batch in preds]
        epoch_loss = torch.stack(batch_loss).mean()
        batch_acc = [batch["acc"] for batch in preds]
        epoch_acc = torch.stack(batch_acc).mean()
        return {"loss": epoch_loss, "acc": epoch_acc}

    def log(self, epoch, result):
        print(
            f"Epoch [{epoch}], Loss [{result['loss']}], Accuracy [{result['acc']}]\n")


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item()/len(preds))


def evaluate(model, valid_loader):
    preds = [model.validate(batch) for batch in valid_loader]
    return model.compute_metric(preds)


def fit(epochs, learning_rate, model, train_loader, valid_loader, opt_func=torch.optim.SGD):
    """
    for epochs:

        # Train Model
        for batches in training set:
            Generate Predictions
            Calculate loss
            Compute Gradients
            Update weights
            Reset gradients

        # Validate Model
        for batches in validation set:
            Generate Predictions
            Calculate loss
            Calculate metrics (accuracy, etc.)

        Calculate average validation loss and metrics

        Log epoch, loss and metrics
    """

    optimizer = opt_func(model.parameters(), learning_rate)
    history = []

    for epoch in range(epochs):

        # Training model by batches
        for batch in train_loader:
            loss = model.train(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Validating model
        result = evaluate(model, valid_loader)
        model.log(epoch, result)
        history.append(result)

    return history


if __name__ == "__main__":
    # Download entire dataset and testing dataset
    dataset = MNIST(root='data/', download=True)
    test_dataset = MNIST(root='data/', train=False)

    # Converts images from Pillow to tensors
    dataset = MNIST(root='data/', train=True, transform=transforms.ToTensor())

    # Define hyperparameters
    train_dataset_size = int(0.8 * len(dataset))
    validation_dataset_size = int(0.2 * len(dataset))
    epochs = 20
    learning_rate = 0.001
    batch_size = 128
    input_size = 28*28
    num_classes = 10

    # Split dataset into training and validation datasets
    train_ds, val_ds = random_split(
        dataset, [train_dataset_size, validation_dataset_size])

    # Convert datasets into dataloaders
    train_loader = DataLoader(train_ds, batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size)

    # Define model
    model = MNISTModel()

    history = fit(epochs, learning_rate, model, train_loader, val_loader)
    # print(history)

    # Save model weights and biases
    torch.save(model.state_dict(), 'mnist-logistic.pth')
