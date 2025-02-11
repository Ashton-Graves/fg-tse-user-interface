import torch
import torch.nn as nn
import numpy as np
# from .preemphasis import FIRFilter
from typing import Literal

class FGMasking(nn.Module):
    def __init__(self, sr: int, perceptual: bool = False,
                 segment_size: int = 4000, thresh: float = 0.1,
                 aggregation_type: Literal["mean", "max", "perc75", "dBFS"] = 'mean',
                 thresh_std: float = None,
                 probabilistic: bool = False,
                 device='cuda'):
        super(FGMasking, self).__init__()
        self.perceptual = perceptual
        self.segment_size = segment_size
        self.thresh = thresh
        self.aggregation_type = aggregation_type

        self.probilistic = probabilistic
        if probabilistic:
            assert thresh_std is not None
            self.thresh_std = thresh_std

        # if perceptual:
        #     self.prefilter = FIRFilter('aw', fs=sr)
        #     self.prefilter.to(device)

    def forward(self, est: torch.Tensor, gt: torch.Tensor):
        B, C, T = est.shape
        segment_size = self.segment_size

        assert T % segment_size == 0
        num_chunks = T // segment_size

        residual = est - gt

        if self.perceptual:
            residual  = self.prefilter(residual)

        residual = residual.reshape(B, C, num_chunks, segment_size).flatten(0,2)

        # Calculate criterion
        if self.aggregation_type == "dBFS":
            criterion = 10 * torch.log10((residual**2).mean(1))
        elif self.aggregation_type == "mean":
            criterion = torch.abs(residual).mean(1)
        elif self.aggregation_type == "max":
            criterion = torch.abs(residual).max(1).values
        elif self.aggregation_type == "perc75":
            criterion = torch.quantile(torch.abs(residual), q=0.75, dim=1)
        else:
            assert 0, f"Unsupported aggregation type {self.aggregation_type}!"

        # Compare with threshold
        if self.probilistic:
            threshold = np.random.normal(loc=self.thresh, scale=self.thresh_std)
        else:
            threshold = self.thresh

        mask = criterion > threshold

        mask = mask.reshape(B, C, num_chunks, 1)
        mask = torch.tile(mask, (1, 1, 1, segment_size))
        mask = mask.reshape(B, C, num_chunks * segment_size)

        return mask.float()
