import torch
import torch_geometric
from torch_geometric.data import Data


def image_to_graph(
    image: torch.Tensor, conv2d: torch.nn.Conv2d | None = None
) -> torch_geometric.data.Data:
    """
    Converts an image tensor to a PyTorch Geometric Data object.
    COMPLETE

    Arguments:
    ----------
    image : torch.Tensor
        Image tensor of shape (C, H, W).
    conv2d : torch.nn.Conv2d, optional
        Conv2d layer to simulate, by default None
        Is used to determine the size of the receptive field.

    Returns:
    --------
    torch_geometric.data.Data
        Graph representation of the image.
    """
    H, W = image.size(1), image.size(2)
    x = image.view(image.size(0), -1).T

    edges = [
        (
            W * i + j,
            W * (i + di) + j + dj,
            conv2d.weight[:, :, di + 1, dj + 1],
        )  # for now, suppose cout = 1. di+1 and dj+1 because we assume 3x3 kernel
        for i in range(H)
        for j in range(W)
        for di, dj in [
            (1, 0),
            (-1, 0),
            (0, -1),
            (0, 1),
            (0, 0),
            (1, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
        ]
        if 0 <= i + di < H and 0 <= j + dj < W
    ]

    node0 = [edge[0] for edge in edges]
    node1 = [edge[1] for edge in edges]

    edge_index = [node0, node1]
    edge_index = torch.tensor(edge_index)

    edge_attr = [edge[2] for edge in edges]
    edge_attr = torch.stack(edge_attr)

    # Assumptions (remove it for the bonus)
    assert image.dim() == 3, f"Expected 3D tensor, got {image.dim()}D tensor."
    if conv2d is not None:
        assert conv2d.padding[0] == conv2d.padding[1] == 1, (
            "Expected padding of 1 on both sides."
        )
        assert conv2d.kernel_size[0] == conv2d.kernel_size[1] == 3, (
            "Expected kernel size of 3x3."
        )
        assert conv2d.stride[0] == conv2d.stride[1] == 1, "Expected stride of 1."

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def graph_to_image(
    data: torch.Tensor, height: int, width: int, conv2d: torch.nn.Conv2d | None = None
) -> torch.Tensor:
    """
    Converts a graph representation of an image to an image tensor.

    Arguments:
    ----------
    data : torch.Tensor
        Graph data representation of the image.
    height : int
        Height of the image.
    width : int
        Width of the image.
    conv2d : torch.nn.Conv2d, optional
        Conv2d layer to simulate, by default None

    Returns:
    --------
    torch.Tensor
        Image tensor of shape (C, H, W).
    """

    assert data.dim() == 2, f"Expected 2D tensor, got {data.dim()}D tensor."
    if conv2d is not None:
        assert conv2d.padding[0] == conv2d.padding[1] == 1, (
            "Expected padding of 1 on both sides."
        )
        assert conv2d.kernel_size[0] == conv2d.kernel_size[1] == 3, (
            "Expected kernel size of 3x3."
        )
        assert conv2d.stride[0] == conv2d.stride[1] == 1, "Expected stride of 1."

    return data.T.view(data.size(1), height, width)


class Conv2dMessagePassing(torch_geometric.nn.MessagePassing):
    """
    A Message Passing layer that simulates a given Conv2d layer.
    """

    def __init__(self, conv2d: torch.nn.Conv2d):
        super().__init__(aggr="add")

    def forward(self, data):
        self.edge_index = data.edge_index

        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return out

    def message(self, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Computes the message to be passed for each edge.
        FOR EACH EDGE E = (U, V) IN THE GRAPH INDEXED BY I,
        THE MESSAGE TROUGH THE EDGE E (IE FROM NODE U TO NODE V)
        SHOULD BE RETURNED AS THE I-TH LINE OF THE OUTPUT TENSOR.
        (The message is phi(u, v, e) in the formalism.)
        To do this you can access the features of the source node
        in x_j[i] and the attributes of the edge in edge_attr[i].

        Arguments:
        ----------
        x_j : torch.Tensor
            The features of the souce node for each edge (of size E x in_channels).
        edge_attr : torch.Tensor
            The attributes of the edge (of size E x edge_attr_dim).

        Returns:
        --------
        torch.Tensor
            The message to be passed for each edge (of size COMPLETE)
        """

        return torch.bmm(edge_attr, x_j.unsqueeze(-1)).squeeze(-1)
