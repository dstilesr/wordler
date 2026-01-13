import torch
from pathlib import Path
from torch import nn
from loguru import logger

from .self_attn import SelfAttention
from .settings import ActorModelSettings
from ..constants import TOKEN_MAP, RESULT_MAP
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
            num_embeddings=len(TOKEN_MAP) + 1,
            embedding_dim=settings.embedding_dim,
            dtype=dtype,
        )
        self.feedback_embedding = nn.Embedding(
            num_embeddings=len(RESULT_MAP) + 1,
            embedding_dim=settings.embedding_dim,
            dtype=dtype,
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

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir: Path,
        vocabulary_size: int,
        dtype: torch.dtype = torch.float32,
    ) -> "ActorModel":
        """
        Load an ActorModel from a checkpoint directory.

        The checkpoint directory must contain:
        - model.pt: Saved model state dict
        - settings.json: Model architecture settings

        :param checkpoint_dir: Path to checkpoint directory
        :param vocabulary_size: Vocabulary size for the model
        :param dtype: Data type for model parameters (default: float32)
        :return: Loaded ActorModel instance
        :raises FileNotFoundError: If checkpoint directory or required files don't exist
        :raises ValueError: If checkpoint path is not a directory or vocabulary size mismatch
        """
        if not checkpoint_dir.exists():
            raise FileNotFoundError(
                f"Checkpoint directory not found: {checkpoint_dir}"
            )

        if not checkpoint_dir.is_dir():
            raise ValueError(
                f"Checkpoint path must be a directory: {checkpoint_dir}"
            )

        model_file = checkpoint_dir / "model.pt"
        settings_file = checkpoint_dir / "settings.json"

        if not model_file.exists():
            raise FileNotFoundError(
                f"Model file not found in checkpoint: {model_file}"
            )

        if not settings_file.exists():
            raise FileNotFoundError(
                f"Settings file not found in checkpoint: {settings_file}"
            )

        logger.info(
            "Loading model from checkpoint directory: {}", checkpoint_dir
        )

        # Load settings from JSON file using Pydantic
        settings_json = settings_file.read_text()
        settings = ActorModelSettings.model_validate_json(settings_json)
        logger.info("Loaded model settings from {}", settings_file)
        logger.debug("Model settings: {}", settings)

        # Create model with loaded architecture settings
        model = cls(
            settings=settings, vocabulary_size=vocabulary_size, dtype=dtype
        )

        # Load state dict
        state_dict = torch.load(
            model_file, map_location="cpu", weights_only=True
        )
        model.load_state_dict(state_dict)

        # Validate vocabulary size by checking output layer
        loaded_vocab_size = model.out_layer.out_features
        if loaded_vocab_size != vocabulary_size:
            raise ValueError(
                f"Model vocabulary size mismatch: loaded model has "
                f"{loaded_vocab_size}, but expected {vocabulary_size}"
            )

        logger.info(
            "Successfully loaded model with vocabulary size {}", vocabulary_size
        )
        return model

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
