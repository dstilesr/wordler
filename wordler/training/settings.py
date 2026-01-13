from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class TrainSettings(BaseSettings):
    """
    Settings for training the model.
    """

    learning_rate: float = Field(default=0.001, le=1.0, gt=0.0)
    discount: float = Field(default=1.0, le=1.0, ge=0.0)
    temperature: float = Field(default=2.0, gt=0.0)

    model_config = SettingsConfigDict(env_prefix="TRAINING_")
