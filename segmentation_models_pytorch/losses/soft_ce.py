from typing import Optional
from torch import nn, Tensor
import torch
import torch.nn.functional as F

from segmentation_models_pytorch.decoders.unet.model import Unet
from ._functional import label_smoothed_nll_loss
#from losses._functional import label_smoothed_nll_loss

__all__ = ["SoftCrossEntropyLoss"]


class SoftCrossEntropyLoss(nn.Module):
    __constants__ = ["reduction", "ignore_index", "smooth_factor"]

    def __init__(
            self,
            reduction: str = "mean",
            smooth_factor: Optional[float] = None,
            ignore_index: Optional[int] = -100,
            dim: int = 1,
            topk: float = 1,
    ):
        """Drop-in replacement for torch.nn.CrossEntropyLoss with label_smoothing

        Args:
            smooth_factor: Factor to smooth target (e.g. if smooth_factor=0.1 then [1, 0, 0] -> [0.9, 0.05, 0.05])

        Shape
             - **y_pred** - torch.Tensor of shape (N, C, H, W)
             - **y_true** - torch.Tensor of shape (N, H, W)

        Reference
            https://github.com/BloodAxe/pytorch-toolbelt
        """
        super().__init__()
        self.smooth_factor = smooth_factor
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.dim = dim
        self.topk = topk

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        log_prob = F.log_softmax(y_pred, dim=self.dim)
        return label_smoothed_nll_loss(
            log_prob,
            y_true,
            epsilon=self.smooth_factor,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            dim=self.dim,
            topk=self.topk
        )


if __name__ == '__main__':
    model = Unet(encoder_name='resnet18', classes=4)
    output = model(torch.randn(2, 16, 1024))
    binary_target = torch.randint(0, 1, (2, 4, 1024))
    target = torch.randint(0, 4, (2, 1024))
    loss = SoftCrossEntropyLoss(smooth_factor=0, topk=0.1)
    print(output.size())
    print(loss(output, target))
