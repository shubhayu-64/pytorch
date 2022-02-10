import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn.functional as func


def fetch_data():
    # Input (temp, rainfall, humidity)
    inputs = np.array([[73, 67, 43],
                       [91, 88, 64],
                       [87, 134, 58],
                       [102, 43, 37],
                       [69, 96, 70],
                       [74, 66, 43],
                       [91, 87, 65],
                       [88, 134, 59],
                       [101, 44, 37],
                       [68, 96, 71],
                       [73, 66, 44],
                       [92, 87, 64],
                       [87, 135, 57],
                       [103, 43, 36],
                       [68, 97, 70]],
                      dtype='float32')

    # Targets (apples, oranges)
    targets = np.array([[56, 70],
                        [81, 101],
                        [119, 133],
                        [22, 37],
                        [103, 119],
                        [57, 69],
                        [80, 102],
                        [118, 132],
                        [21, 38],
                        [104, 118],
                        [57, 69],
                        [82, 100],
                        [118, 134],
                        [20, 38],
                        [102, 120]],
                       dtype='float32')

    inputs = torch.from_numpy(inputs)
    targets = torch.from_numpy(targets)
    return inputs, targets


def fit(train_dl, epochs, model, loss_fn, opt):

    for curr_epoch in range(epochs):

        for xb, yb in train_dl:
            # Predict based on inputs
            pred = model(xb)

            # Calculate the loss
            loss = loss_fn(pred, yb)

            # Compute gradients for Gradient Descent
            loss.backward()

            # Optimize the weights and biases based on gradient
            opt.step()

            # Reset gradients to zero
            opt.zero_grad()

            # Print the progress
            if (curr_epoch+1) % 10 == 0:
                print('Epoch [{}/{}], Loss: {:.4f}'.format(curr_epoch +
                      1, epochs, loss.item()))
    return model


if __name__ == "__main__":
    print("Starting..\n")

    learning_rate = 1e-5

    # Fetch Data
    inputs, targets = fetch_data()

    # Convert raw data into datasets
    train_ds = TensorDataset(inputs, targets)

    # Build Dataloaders for to break dataset into batches
    batch_size = 5
    train_dl = DataLoader(train_ds, batch_size, shuffle=True)

    # Define a model to train
    model = nn.Linear(3, 2)

    # Define a loss function
    loss_fn = func.mse_loss

    # Define optimizer
    opt = torch.optim.SGD(model.parameters(), learning_rate)

    # Train model
    model = fit(train_dl, 200, model, loss_fn, opt)

    # Generate predictions
    preds = model(inputs)
    print(preds)
    print(targets)
