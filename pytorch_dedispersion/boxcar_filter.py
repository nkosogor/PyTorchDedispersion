from typing import Sequence
import torch

class BoxcarFilter:
    def __init__(self, data_tensor: torch.Tensor) -> None:
        """
        Initialize BoxcarFilter.

        Args:
            data_tensor (torch.Tensor): The dedispersed data tensor.
        """
        self.data_tensor = data_tensor

    def apply_boxcar(self, widths: Sequence[int]) -> torch.Tensor:
        """
        Apply boxcar filtering with different widths.

        Args:
            widths (list[int]): List of boxcar widths.

        Returns:
            torch.Tensor: Boxcar filtered data for each width.
        """
        boxcar_data = []
        for width in widths:
            kernel = torch.ones((width,), dtype=torch.float32, device=self.data_tensor.device) / width
            boxcar_filtered = torch.nn.functional.conv1d(self.data_tensor.unsqueeze(1), kernel.unsqueeze(0).unsqueeze(1), padding='same').squeeze(1)
            boxcar_data.append(boxcar_filtered)
        return torch.stack(boxcar_data)
