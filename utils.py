import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models
from tqdm import tqdm

import wandb


def compute_grad_norm(model: nn.Module) -> float:
    total_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            total_norm += param.grad.data.norm(2).item() ** 2
    return total_norm**0.5


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
    data, targets = data.to(device), targets.to(device)
    features = model(data)

    features_dataset = TensorDataset(
        features.detach().clone(), targets.detach().clone()
    )  # FREE THE GRADIENTS before returning the dataset

    return features_dataset


def train(
    model, train_dataloader, test_dataloader, loss_fn, optimizer, n_epochs, device
):
    train_loss_history = []
    test_loss_history = []
    accuracy_history = []

    model.to(device)

    for i in tqdm(range(n_epochs)):
        model.train()
        train_loss = 0.0
        test_loss = 0.0
        correct = 0.0
        for features, targets in train_dataloader:
            features, targets = features.to(device), targets.to(device)
            optimizer.zero_grad()
            y_pred = model(features)
            loss = loss_fn(y_pred, targets)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        train_loss /= len(train_dataloader)

        with torch.no_grad():
            model.eval()
            for features, targets in test_dataloader:
                features, targets = features.to(device), targets.to(device)
                y_pred = model(features)
                loss = loss_fn(y_pred, targets)
                test_loss += loss.item()
                correct += (torch.argmax(y_pred, dim=1) == targets).sum().item()

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


def merge_models(resnet1, resnet2, device):
    """resnet1 += resnet2"""
    resnet1.to(device)
    resnet2.to(device)
    for child1, child2 in zip(resnet1.children(), resnet2.children()):
        if isinstance(child1, nn.Conv2d):
            child1.weight.data += child2.weight.data.clone()
        else:
            merge_models(child1, child2, device)


def merge_lora_conv(
    downconv: nn.Conv2d,
    upconv: nn.Conv2d,
    Cin: int,
    Cout: int,
    padding: int,
    stride: int,
    kernel_size: int,
) -> nn.Conv2d:
    """
    Merge two convolutions into a single convolution. The second
    Args:
        downconv: The first convolution (r x Cin x kernel_size x kernel_size)
        upconv: The second convolution (Cout x r x 1 x 1)
        Cin: Number of input channels
        Cout: Number of output channels
        padding: Padding of the downConv which is the same as the Conv in the pre-lora resnet
        stride: stride of the downConv which is the same as the Conv in the pre-lora resnet
        r: Rank of the factorized convolution
        kernel_size: Size of the convolution kernel
    """
    merged_conv = nn.Conv2d(
        Cin, Cout, kernel_size=kernel_size, padding=padding, stride=stride, bias=False
    )

    with torch.no_grad():
        # Merge weights via einsum (conv2 is 1x1, so weight[:, :, 0, 0] is our channel-mixing matrix)
        # shape of conv1.weight: (r, Cin, kH, kW)
        # shape of conv2.weight: (Cout, r, 1, 1) -> can be viewed as (Cout, r)
        merged_conv.weight.copy_(
            torch.einsum("or,rchw->ochw", upconv.weight[:, :, 0, 0], downconv.weight)
        )

    return merged_conv


class ConvLora(nn.Module):
    def __init__(self, frozen_conv, r=2, alpha=2):
        super().__init__()
        self.alpha = alpha
        self.r = r
        self.downConv = nn.Conv2d(
            frozen_conv.in_channels,
            r,
            frozen_conv.kernel_size,
            frozen_conv.stride,
            frozen_conv.padding,
        )
        self.upConv = nn.Conv2d(
            r, frozen_conv.out_channels, kernel_size=1, padding=0, stride=1
        )

    def forward(self, x):
        lora_x = self.downConv(x)
        lora_x = self.upConv(lora_x)
        return self.alpha / self.r * lora_x


def inflate_lora(lora_resnet):
    for name, child in lora_resnet.named_children():
        if isinstance(child, ConvLora):
            layer = merge_lora_conv(
                child.downConv,
                child.upConv,
                child.downConv.in_channels,
                child.upConv.out_channels,
                child.downConv.padding,
                child.downConv.stride,
                child.downConv.kernel_size[0],
            )
            setattr(lora_resnet, name, layer)
        else:
            inflate_lora(child)


def train_lora(
    pretrained_model,
    lora_model,
    test_dataloader,
    train_dataloader,
    loss_fn,
    optimizer,
    n_epochs,
    device,
):
    wandb.init(project="lora")
    wandb.Settings(quiet=True)

    train_loss_history = []
    test_loss_history = []
    accuracy_history = []

    pretrained_model.to(device)
    lora_model.to(device)

    with torch.no_grad():
        lora_model.eval()
        pretrained_model.eval()
        correct = 0.0
        for features, targets in test_dataloader:
            features, targets = features.to(device), targets.to(device)
            y_pred = lora_model(features) + pretrained_model(features)
            correct += (torch.argmax(y_pred, dim=1) == targets).sum().item()
        accuracy = correct / len(test_dataloader.dataset)

        print(f"Initial accuracy: {accuracy:.4f}")

    for i in tqdm(range(n_epochs)):
        lora_model.train()
        train_loss = 0.0
        test_loss = 0.0
        correct = 0.0
        for features, targets in train_dataloader:
            features, targets = features.to(device), targets.to(device)
            optimizer.zero_grad()
            y_pred = lora_model(features) + pretrained_model(features)
            loss = loss_fn(y_pred, targets)
            loss.backward()
            wandb.log({"grad_norm": compute_grad_norm(lora_model)})

            # Clipping gradients because their monitoring showed huge spikes
            # torch.nn.utils.clip_grad_norm_(lora_model.parameters(), max_norm=15.0)
            train_loss += loss.item()
            optimizer.step()
        train_loss /= len(train_dataloader)

        with torch.no_grad():
            lora_model.eval()
            pretrained_model.eval()
            for features, targets in test_dataloader:
                features, targets = features.to(device), targets.to(device)
                y_pred = lora_model(features) + pretrained_model(features)
                loss = loss_fn(y_pred, targets)
                test_loss += loss.item()
                correct += (torch.argmax(y_pred, dim=1) == targets).sum().item()
            test_loss /= len(test_dataloader)

        accuracy = correct / len(test_dataloader.dataset)

        train_loss_history.append(train_loss)
        test_loss_history.append(test_loss)
        accuracy_history.append(accuracy)

        wandb.log(
            {
                "train_loss": train_loss,
                "test_loss": test_loss,
                "accuracy": accuracy,
            }
        )

        if i % 5 == 0:
            print(
                f"Epoch {i + 1}, train loss: {train_loss:.4f}, test loss: {test_loss:.4f}, accuracy: {accuracy:.4f}"
            )

    wandb.finish()
    return train_loss_history, test_loss_history, accuracy_history


def build_lora_resnet(module, r=4, alpha=4):
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            # Freeze the original conv's parameters
            child.requires_grad_(False)
            # Create a ConvLora layer using the original conv as frozen_weights
            new_layer = ConvLora(
                child,
                r=r,
                alpha=alpha,
            )
            setattr(module, name, new_layer)
        else:
            # Recursively replace in child modules
            build_lora_resnet(child, r=r, alpha=alpha)
