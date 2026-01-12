import torch
from torch import nn

from .self_attn import SelfAttention
from .settings import ActorModelSettings
from .positional_encode import PositionalEncoding


class ActorModel(nn.Module):
    """
    Actor model for the Wordle game.
    """

    def __init__(
        self,
        settings: ActorModelSettings,
        vocabulary_size: int,
        dtype: torch.dtype = torch.float32,
    ):
        """
        Initialize the actor model.
        :param settings:
        """
        super().__init__()
        self.num_layers = settings.num_seq_layers

        # Initialize layers
        self.letter_embedding = nn.Embedding(
            num_embeddings=28, embedding_dim=settings.embedding_dim, dtype=dtype
        )
        self.feedback_embedding = nn.Embedding(
            num_embeddings=5, embedding_dim=settings.embedding_dim, dtype=dtype
        )
        self.pos_encoder = PositionalEncoding(
            embedding_dim=2 * settings.embedding_dim, dtype=dtype
        )
        self.reducer = nn.Linear(
            in_features=2 * settings.embedding_dim,
            out_features=settings.embedding_dim,
            dtype=dtype,
        )

        self.sequence_layers = nn.ModuleList(
            [
                SelfAttention(
                    input_dim=settings.embedding_dim,
                    output_dim=settings.embedding_dim,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.dropouts = nn.ModuleList(
            [nn.Dropout(p=settings.dropout) for _ in range(self.num_layers)]
        )
        self.fcs = nn.ModuleList(
            [
                nn.Linear(
                    in_features=settings.embedding_dim,
                    out_features=settings.embedding_dim,
                    dtype=dtype,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.activations = nn.ModuleList(
            [nn.ReLU() for _ in range(self.num_layers)]
        )
        self.out_dropout = nn.Dropout(p=settings.dropout)
        self.out_layer = nn.Linear(
            in_features=settings.embedding_dim,
            out_features=vocabulary_size,
            dtype=dtype,
        )

    def forward(
        self,
        letters_seq: torch.Tensor,
        results_seq: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Predict the next guess.
        :param letters_seq: Shape (batch_size, seq_len)
        :param results_seq: Shape (batch_size, seq_len)
        :param mask: Optional mask for the attention scores. Shape
            (batch_size, seq_len).
        :return:
        """
        # Embed the input sequences -> shape: (batch_size, seq_len, embedding_dim)
        letters_embed = self.letter_embedding(letters_seq)
        results_embed = self.feedback_embedding(results_seq)

        #: Shape: (batch_size, seq_len, 2 * embedding_dim)
        embed = torch.concat((letters_embed, results_embed), dim=-1)
        embed = self.pos_encoder(embed)

        #: Shape: (batch_size, seq_len, embedding_dim)
        x = self.reducer(embed)

        for i in range(self.num_layers):
            x_new = self.sequence_layers[i](x, mask)
            x_new = self.dropouts[i](x_new)
            x_new = self.fcs[i](x_new)

            x_new = self.activations[i](x_new)
            x = x + x_new  #: Residual connection

        #: Shape: (batch_size, embedding_dim)
        x = x[:, 0, :]

        #: Shape: (batch_size, vocabulary_size)
        x = self.out_dropout(x)
        x = self.out_layer(x)
        return x  #: Return logits
