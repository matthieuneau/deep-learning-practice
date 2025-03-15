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
    test_dataloader,
    train_dataloader,
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


def merge_models(resnet1, resnet2):
    """resnet1 += resnet2"""
    for child1, child2 in zip(resnet1.children(), resnet2.children()):
        if isinstance(child1, nn.Conv2d):
            child1.weight.data += child2.weight.data.clone()
        else:
            merge_models(child1, child2)
