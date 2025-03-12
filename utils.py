import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models
import torch


def precompute_features(
    model: models.ResNet, 
    dataset: torch.utils.data.Dataset, 
    device: torch.device
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

    one_hot_targets = F.one_hot(targets.long(), num_classes=2).to(torch.float)

    features_dataset = TensorDataset(features.detach().clone(), one_hot_targets.detach().clone())   # FREE THE GRADIENTS before returning the dataset

    return features_dataset