import torch


class PositionalEncoding(torch.nn.Module):
    """
    Positional encoding layer.
    """

    def __init__(
            self,
            embedding_dim: int,
            max_len: int = 60,
            scale: float = 10000.0,
            dtype: torch.dtype = torch.float32):
        """
        Initialize the positional encoding layer.
        :param embedding_dim: Dimension of the model.
        :param max_len: Maximum length of the input sequences.
        :param scale: Scaling factor for the positional encodings.
        :param dtype: Data type for the model parameters.
        """
        super().__init__()
        self.d_model = embedding_dim

        # Create a long enough P
        position = torch.arange(0, max_len, dtype=dtype).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0,
            embedding_dim,
            2,
            dtype=dtype) * (-torch.log(torch.tensor(scale)) / embedding_dim))
        pe = torch.zeros((max_len, embedding_dim), dtype=dtype)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(
            self,
            x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the positional encoding layer.
        :param x: Input sequence. Shape (batch_size, seq_len, d_model).
        :return:
        """
        x = x * torch.sqrt(torch.tensor(self.d_model, dtype=x.dtype))
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return x
