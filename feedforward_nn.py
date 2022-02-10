import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import random_split
from torchvision.datasets import MNIST


class MnistModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        # Hidden layer
        self.hidden = nn.Linear(input_size, hidden_size)
        # Output layer
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, batch):
        batch = batch.view(batch.size(0), -1)
        hidden_output = self.hidden(batch)
        hidden_output = F.relu(hidden_output)
        output = self.output(hidden_output)
        return output

    def train(self, batch):
        image, label = batch
        output = self(image)
        loss = F.cross_entropy(output, label)
        return loss

    def validate(self, batch):
        image, label = batch
        output = self(image)
        loss = F.cross_entropy(output, label)
        acc = accuracy(output, label)
        return {"loss": loss, "acc": acc}

    def compute_metrics(self, preds):
        batch_loss = [batch["loss"] for batch in preds]
        epoch_loss = torch.stack(batch_loss).mean()
        batch_acc = [batch["acc"] for batch in preds]
        epoch_acc = torch.stack(batch_acc).mean()
        return {"loss": epoch_loss, "acc": epoch_acc}

    def log(self, epoch, result):
        print(
            f"Epoch [{epoch}], Loss [{result['loss']}], Accuracy [{result['acc']}]\n")


class deviceDataLoader():
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        for batch in self.dl:
            yield to_device(batch, self.device)

    def __len__(self):
        print(len(self.dl))
        return len(self.dl)


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


def evaluate(model, valid_dl):
    preds = [model.validate(batch) for batch in valid_dl]
    return model.compute_metrics(preds)


def fit(epochs, learning_rate, model, train_dl, valid_dl, momentum, opt_func=torch.optim.SGD):
    optimizer = opt_func(model.parameters(), learning_rate, momentum=momentum)
    history = []

    for epoch in range(epochs):
        for batch in train_dl:
            loss = model.train(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        result = evaluate(model, valid_dl)
        model.log(epoch, result)
        history.append(result)

    return history


if __name__ == "__main__":
    print("Starting...\n")

    # Fetch dataset
    dataset = MNIST(root="data/", download=True,
                    transform=transforms.ToTensor())
    test_dataset = MNIST(root="data/", train=False,
                         transform=transforms.ToTensor())

    # Define hyperparameters
    train_dataset_size = int(0.8 * len(dataset))
    validation_dataset_size = int(0.2 * len(dataset))
    epochs = 10
    learning_rate = 0.2
    batch_size = 128
    input_size = 28*28
    hidden_size = 512
    num_classes = 10
    momentum = 0.9
    device = get_default_device()

    # Split dataset into training and validation datasets
    train_ds, valid_ds = random_split(
        dataset, [train_dataset_size, validation_dataset_size])

    # Convert datasets into dataloaders
    train_dl = DataLoader(train_ds, batch_size, shuffle=True,
                          num_workers=4, pin_memory=True)
    valid_dl = DataLoader(valid_ds, batch_size*2,
                          num_workers=4, pin_memory=True)

    train_dl = deviceDataLoader(train_dl, device)
    valid_dl = deviceDataLoader(valid_dl, device)

    # Model declaration
    model = MnistModel(input_size, hidden_size, num_classes)
    to_device(model, device)

    history = fit(epochs, learning_rate, model, train_dl, valid_dl, momentum)

    # # Plot losses
    losses = np.array([x['loss'].item() for x in history])
    plt.plot(losses, '-x')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Loss vs. No. of epochs')
    plt.show()

    # Plot accuracy
    accuracies = np.array([x['acc'].item() for x in history])
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs')
    plt.show()

    test_loader = deviceDataLoader(DataLoader(
        test_dataset, batch_size=256, num_workers=4, pin_memory=True), device)
    result = evaluate(model, test_loader)
    print(result)

    # Save model weights and biases
    torch.save(model.state_dict(), 'mnist-feedforwardnn.pth')
