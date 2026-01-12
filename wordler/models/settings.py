from typing import Literal
from pydantic import BaseModel, Field


class ActorModelSettings(BaseModel):
    """
    Settings for the Actor Model.
    """

    embedding_dim: int = Field(
        default=128,
        description="Dimension of the embedding layers.",
        gt=0,
    )
    hidden_dim: int = Field(
        default=256,
        description="Dimension of the hidden layers.",
        gt=0,
    )
    num_seq_layers: int = Field(
        default=2,
        description="Number of sequence processing layers.",
        gt=0,
    )
    dropout: float = Field(
        default=0.1,
        description="Dropout rate for the model.",
        ge=0.0,
        lt=1.0,
    )
