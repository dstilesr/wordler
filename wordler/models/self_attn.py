import math
import torch
from torch import nn
import torch.functional as F


class SelfAttention(nn.Module):
    """
    Scaled dot product self-attention layer.
    """

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            dtype: torch.dtype = torch.float32):
        """
        Initialize the self-attention layer.
        :param input_dim:
        :param output_dim:
        :param dtype: Data type for the model parameters.
        """
        super().__init__()

        self.query = nn.Linear(
            input_dim,
            output_dim,
            bias=False,
            dtype=dtype
        )
        self.key = nn.Linear(
            input_dim,
            output_dim,
            bias=False,
            dtype=dtype
        )
        self.value = nn.Linear(
            input_dim,
            output_dim,
            bias=False,
            dtype=dtype
        )
        self.softmax = nn.Softmax(dim=-1)
        self.output_dim = output_dim

    def forward(
            self,
            x: torch.Tensor,
            mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Forward pass for the self-attention layer.
        :param x: Input sequence. Shape (batch_size, seq_len, input_dim).
        :param mask: Optional mask for the attention scores. Shape
            (batch_size, seq_len).
        :return:
        """
        queries = self.query(x)
        keys: torch.Tensor = self.key(x)
        values = self.value(x)

        scores = (
            torch.matmul(queries, keys.transpose(1, 2))
            / math.sqrt(self.output_dim)
        )  #: Shape (batch_size, seq_len, seq_len)

        if mask is not None:
            mask_expand = mask.unsqueeze(1)  #: Shape (batch_size, 1, seq_len)
            scores = (
                scores
                + torch.where(mask_expand.eq(0), -1e9, 0).to(scores.dtype)
            )
        weights = self.softmax(scores)

        out_seq = torch.matmul(
            weights,
            values
        )  #: Shape (batch_size, seq_len, output_dim)
        return out_seq
