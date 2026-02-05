import torch


def batch_pearson_images(x: torch.Tensor, y: torch.Tensor, average_over_channels: bool = True,
                         average_over_batches: bool = True, invert: bool = False) -> torch.Tensor:
    """
    Compute batch-wise Pearson Correlation Coefficient between tensors x and y.
    Shape: (N, C, H, W)
    """
    # Ensure the shapes match
    assert x.shape == y.shape, "Input tensors must have the same shape"

    x = torch.abs(x)
    y = torch.abs(y)

    # Mean normalization
    x_mean = torch.mean(x, dim=(2, 3), keepdim=True)
    y_mean = torch.mean(y, dim=(2, 3), keepdim=True)
    x_std = torch.std(x, dim=(2, 3), keepdim=True)
    y_std = torch.std(y, dim=(2, 3), keepdim=True)

    # Compute covariance and standard deviations
    cov = torch.mean((x - x_mean) * (y - y_mean), dim=(2, 3), keepdim=True)

    # Compute Pearson correlation, avoid division by zero
    corr = cov / (x_std * y_std + 1e-8)

    # Average over channels, and then over batch
    if average_over_channels:
        corr = torch.mean(corr, dim=1, keepdim=True)
    if average_over_batches:
        corr = torch.mean(corr)

    if invert:
        corr = 1 - corr

    return corr


class PearsonImageLoss(torch.nn.Module):

    def __init__(self, average_over_channels: bool = True, average_over_batches: bool = True, invert=True,
                 power: int = 1):
        super(PearsonImageLoss, self).__init__()
        self.average_over_channels = average_over_channels
        self.average_over_batches = average_over_batches
        self.invert = invert
        self.power = power
        self.pearson = batch_pearson_images

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pearson_value = self.pearson(
            torch.pow(x, self.power), torch.pow(y, self.power),
            average_over_channels=self.average_over_channels,
            average_over_batches=self.average_over_batches,
            invert=self.invert,
        )
        return pearson_value
