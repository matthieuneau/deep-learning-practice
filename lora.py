# for now, only handle 3x3 conv with padding=stride=1
import torch
import torch.nn as nn
from tqdm import tqdm


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


def build_lora_resnet(module, r=2, alpha=10):
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


def train_lora(
    pretrained_model,
    lora_model,
    train_dataloader,
    test_dataloader,
    loss_fn,
    optimizer,
    n_epochs,
):
    train_loss_history = []
    test_loss_history = []
    accuracy_history = []

    for i in tqdm(range(n_epochs)):
        lora_model.train()
        train_loss = 0
        test_loss = 0
        correct = 0
        for features, targets in train_dataloader:
            optimizer.zero_grad()
            y_pred = lora_model(features) + pretrained_model(features)
            loss = loss_fn(y_pred, targets)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        with torch.no_grad():
            lora_model.eval()
            for features, targets in test_dataloader:
                y_pred = lora_model(features) + pretrained_model(features)
                loss = loss_fn(y_pred, targets)
                test_loss += loss.item()
                correct += (torch.argmax(y_pred, dim=1) == targets).sum()

        accuracy = correct / len(test_dataloader.dataset)

        train_loss_history.append(train_loss)
        test_loss_history.append(test_loss)
        accuracy_history.append(accuracy)

        if i % 2 == 0:
            print(
                f"Epoch {i + 1}, train loss: {train_loss:.4f}, test loss: {test_loss:.4f}, accuracy: {accuracy:.4f}"
            )

    return train_loss_history, test_loss_history, accuracy_history
