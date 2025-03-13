import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models
from tqdm import tqdm


def precompute_features(
    model: models.ResNet, dataset: torch.utils.data.Dataset, device: torch.device
) -> torch.utils.data.Dataset:
    """
    Create a new dataset with the features precomputed by the model.

    If the model is $f \circ g$ where $f$ is the last layer and $g$ is
    the rest of the model, it is not necessary to recompute $g(x)$ at
    each epoch as $g$ is fixed. Hence you can precompute $g(x)$ and
    create a new dataset
    $\mathcal{X}_{\text{train}}' = \{(g(x_n),y_n)\}_{n\leq N_{\text{train}}}$

    Arguments:
    ----------
    model: models.ResNet
        The model used to precompute the features
    dataset: torch.utils.data.Dataset
        The dataset to precompute the features from
    device: torch.device
        The device to use for the computation

    Returns:
    --------
    torch.utils.data.Dataset
        The new dataset with the features precomputed
    """
    # dataset is small so we can use a single batch
    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

    model.to(device)

    data, targets = next(iter(dataloader))
    features = model(data)

    features_dataset = TensorDataset(
        features.detach().clone(), targets.detach().clone()
    )  # FREE THE GRADIENTS before returning the dataset

    return features_dataset


# TODO: Fix
class LastLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(512, 2)

    def forward(self, x):
        return self.linear(x)


def train(model, train_dataloader, test_dataloader, loss_fn, optimizer, n_epochs):
    train_loss_history = []
    test_loss_history = []
    accuracy_history = []

    for i in tqdm(range(n_epochs)):
        model.train()
        train_loss = 0
        test_loss = 0
        correct = 0
        for features, targets in train_dataloader:
            optimizer.zero_grad()
            y_pred = model(features)
            loss = loss_fn(y_pred, targets)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        with torch.no_grad():
            model.eval()
            for features, targets in test_dataloader:
                y_pred = model(features)
                loss = loss_fn(y_pred, targets)
                test_loss += loss.item()
                correct += (torch.argmax(y_pred, dim=1) == targets).sum()

        accuracy = correct / len(test_dataloader.dataset)

        train_loss_history.append(train_loss)
        test_loss_history.append(test_loss)
        accuracy_history.append(accuracy)

        if i % 100 == 0:
            print(
                f"Epoch {i + 1}, train loss: {train_loss:.4f}, test loss: {test_loss:.4f}, accuracy: {accuracy:.4f}"
            )

    return train_loss_history, test_loss_history, accuracy_history


def plot_training(train_loss_history, test_loss_history, accuracy_history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(train_loss_history, label="Train Loss")
    ax1.plot(test_loss_history, label="Test Loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.set_title("Train and Test Loss over Epochs")
    ax1.legend()

    ax2.plot(accuracy_history, label="Accuracy")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy over Epochs")
    ax2.legend()

    plt.show()
