from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class EnvSettings(BaseSettings):
    """
    Settings for the game environment.
    """

    invalid_word_penalty: float = Field(
        -1.0,
        description="Penalty for submitting an invalid word.",
        lt=0.0,
    )
    win_reward: float = Field(
        10.0,
        description="Reward for winning a game.",
        gt=0.0,
    )
    correct_letter_reward: float = Field(
        1.0,
        description="Reward for placing a letter correctly.",
        gt=0.0,
    )
    letter_present_reward: float = Field(
        0.5,
        description="Reward for placing a correct letter in the wrong position.",
        gt=0.0,
    )

    model_config = SettingsConfigDict(env_prefix="ENV_")
