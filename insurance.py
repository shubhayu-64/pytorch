import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.utils.data as data
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn.functional as func


def load_data():
    dataset = pd.read_csv("day.csv", sep=",")
    dataset = dataset[["temp", "casual", "registered", "cnt"]]
    target = torch.tensor(dataset["cnt"].values.astype(np.float32))
    attributes = torch.tensor(dataset.drop(
        "cnt", axis=1).values.astype(np.float32))
    return target, attributes


def Split_Train_Validation(TotalSize: int, TrainSize: float):
    a = np.random.permutation(TotalSize)
    TrainSize = int(TotalSize * TrainSize)
    return a[:TrainSize], a[TrainSize:]


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
                # print('Epoch [{}/{}], Loss: {:.4f}'.format(curr_epoch +1, epochs, loss.item()))
                print()
    return model


if __name__ == "__main__":

    # Define constants
    learning_rate = 1e-6
    batch_size = 15

    # Fetch Data
    target, attribute = load_data()

    # Convert raw data into datasets
    train_ds = TensorDataset(attribute, target)
    # test_ds = TensorDataset(attribute, target)

    # train_set_size = int(len(train_ds) * 0.8)
    # test_set_size = len(train_ds) - train_set_size
    # train_ds, test_ds = data.random_split(
    #     train_ds, [train_set_size, test_set_size])

    # Build Dataloaders for to break dataset into batches
    train_dl = DataLoader(train_ds, batch_size, shuffle=True)

    # Define a model to train
    model = nn.Linear(3, 1)

    # Define a loss function
    loss_fn = func.mse_loss

    # Define optimizer
    opt = torch.optim.SGD(model.parameters(), learning_rate)

    # Train model
    model = fit(train_dl, 10, model, loss_fn, opt)

    # Generate predictions
    # preds = model(test_ds)
    # print(preds)
    # print(target)
