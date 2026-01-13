from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ActorModelSettings(BaseSettings):
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

    model_config = SettingsConfigDict(env_prefix="MODEL_")
