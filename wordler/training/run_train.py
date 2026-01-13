import torch
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from ..models import ActorModel
from ..models.settings import ActorModelSettings
from ..environment.settings import EnvSettings
from .settings import TrainSettings
from .trainer import Trainer, load_vocabulary


def create_settings_from_env() -> tuple[
    ActorModelSettings, EnvSettings, TrainSettings
]:
    """
    Create settings instances from environment variables.

    Environment variables:
    - TRAINING_LEARNING_RATE: Learning rate for optimizer
    - TRAINING_DISCOUNT: Discount factor for returns
    - ENV_WIN_REWARD: Reward for winning
    - ENV_CORRECT_LETTER_REWARD: Reward for correct position
    - ENV_LETTER_PRESENT_REWARD: Reward for correct letter, wrong position
    - MODEL_EMBEDDING_DIM: Embedding dimension
    - MODEL_HIDDEN_DIM: Hidden layer dimension
    - MODEL_NUM_SEQ_LAYERS: Number of sequence layers
    - MODEL_DROPOUT: Dropout rate

    :return: Tuple of (ActorModelSettings, EnvSettings, TrainSettings)
    """
    train_settings = TrainSettings()
    env_settings = EnvSettings()
    actor_settings = ActorModelSettings()

    logger.info("Loaded settings from environment variables")
    logger.debug("Training settings: {}", train_settings)
    logger.debug("Environment settings: {}", env_settings)
    logger.debug("Actor model settings: {}", actor_settings)

    return actor_settings, env_settings, train_settings


def create_or_load_model(
    model_path: Optional[Path],
    actor_settings: ActorModelSettings,
    vocabulary_size: int,
) -> ActorModel:
    """
    Create a new ActorModel or load from a saved checkpoint.

    :param model_path: Path to saved model checkpoint, or None to create new
    :param actor_settings: Settings for model architecture
    :param vocabulary_size: Expected vocabulary size
    :return: ActorModel instance
    :raises ValueError: If loaded model has incorrect vocabulary size
    """
    if model_path is not None:
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        logger.info("Loading model from {}", model_path)

        # Create model with same architecture
        model = ActorModel(
            settings=actor_settings,
            vocabulary_size=vocabulary_size,
            dtype=torch.float32,
        )

        # Load state dict
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)

        # Validate vocabulary size by checking output layer
        loaded_vocab_size = model.out_layer.out_features
        if loaded_vocab_size != vocabulary_size:
            raise ValueError(
                f"Model vocabulary size mismatch: loaded model has "
                f"{loaded_vocab_size}, but current vocabulary has {vocabulary_size}"
            )

        logger.info(
            "Successfully loaded model with vocabulary size {}", vocabulary_size
        )
    else:
        logger.info(
            "Creating new model with vocabulary size {}", vocabulary_size
        )
        model = ActorModel(
            settings=actor_settings,
            vocabulary_size=vocabulary_size,
            dtype=torch.float32,
        )

    return model


def compute_moving_average(
    data: list[float], window_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute moving average of data.

    :param data: Input data
    :param window_size: Size of moving average window
    :return: Tuple of (x_values, moving_averages)
    """
    data_array = np.array(data)
    moving_avg = np.convolve(
        data_array, np.ones(window_size) / window_size, mode="valid"
    )
    # X values start at window_size since that's when we have enough data
    x_values = np.arange(window_size, len(data) + 1)
    return x_values, moving_avg


def save_training_results(
    model: ActorModel,
    wins: list[bool],
    total_rewards: list[float],
    output_path: Path,
) -> Path:
    """
    Save trained model and create plots of training metrics.

    Creates a timestamped folder containing:
    - model.pt: Saved model state dict
    - win_rate.png: Moving average of win rate
    - total_rewards.png: Moving average of total rewards per episode

    :param model: Trained ActorModel
    :param wins: List of boolean values indicating wins
    :param total_rewards: List of total rewards per episode
    :param output_path: Base output directory
    :return: Path to created results folder
    """
    # Create timestamped folder
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M")
    results_dir = output_path / timestamp
    results_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Saving results to {}", results_dir)

    # Save model
    model_path = results_dir / "model.pt"
    torch.save(model.state_dict(), model_path)
    logger.info("Saved model to {}", model_path)

    # Determine window size based on number of episodes
    num_episodes = len(wins)
    if num_episodes < 100:
        window_size = 10
    elif num_episodes < 1000:
        window_size = 50
    elif num_episodes < 10000:
        window_size = 100
    else:
        window_size = 500

    logger.info("Using moving average window size: {}", window_size)

    # Convert wins to numeric for moving average
    wins_numeric = [1.0 if w else 0.0 for w in wins]

    # Plot win rate
    if len(wins_numeric) >= window_size:
        x_win, ma_win = compute_moving_average(wins_numeric, window_size)

        plt.figure(figsize=(12, 6))
        plt.plot(x_win, ma_win, linewidth=2, color="blue")
        plt.xlabel("Episode", fontsize=12)
        plt.ylabel("Win Rate", fontsize=12)
        plt.title(
            f"Win Rate (Moving Average, window={window_size})", fontsize=14
        )
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)

        win_rate_path = results_dir / "win_rate.png"
        plt.savefig(win_rate_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Saved win rate plot to {}", win_rate_path)
    else:
        logger.warning(
            "Not enough episodes ({}) for moving average (need {})",
            num_episodes,
            window_size,
        )

    # Plot total rewards
    if len(total_rewards) >= window_size:
        x_rewards, ma_rewards = compute_moving_average(
            total_rewards, window_size
        )

        plt.figure(figsize=(12, 6))
        plt.plot(x_rewards, ma_rewards, linewidth=2, color="green")
        plt.xlabel("Episode", fontsize=12)
        plt.ylabel("Total Reward", fontsize=12)
        plt.title(
            f"Total Reward per Episode (Moving Average, window={window_size})",
            fontsize=14,
        )
        plt.grid(True, alpha=0.3)

        rewards_path = results_dir / "total_rewards.png"
        plt.savefig(rewards_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Saved total rewards plot to {}", rewards_path)
    else:
        logger.warning(
            "Not enough episodes ({}) for moving average (need {})",
            num_episodes,
            window_size,
        )

    return results_dir


def run_training(
    num_episodes: int,
    output_path: Path,
    model_path: Optional[Path] = None,
) -> Path:
    """
    Main training function that orchestrates the entire training process.

    :param num_episodes: Number of episodes to train for
    :param output_path: Base directory for saving results
    :param model_path: Optional path to load existing model
    :return: Path to results directory
    """
    logger.info("Starting training for {} episodes", num_episodes)

    # Load settings from environment
    actor_settings, env_settings, train_settings = create_settings_from_env()

    # Load vocabulary
    vocabulary = load_vocabulary()
    vocabulary_size = len(vocabulary)

    # Create or load model
    model = create_or_load_model(model_path, actor_settings, vocabulary_size)

    # Create trainer
    trainer = Trainer(
        model=model,
        game_settings=env_settings,
        settings=train_settings,
        vocabulary=vocabulary,
    )

    # Run training
    logger.info("Starting training loop...")
    trainer.train(total_games=num_episodes)
    logger.info("Training complete!")

    # Log final statistics
    final_win_rate = (
        sum(trainer.wins) / len(trainer.wins) if trainer.wins else 0
    )
    avg_reward = np.mean(trainer.total_rewards) if trainer.total_rewards else 0
    logger.info("Final win rate: {:.2%}", final_win_rate)
    logger.info("Average total reward: {:.2f}", avg_reward)

    # Save results
    results_dir = save_training_results(
        model=model,
        wins=trainer.wins,
        total_rewards=trainer.total_rewards,
        output_path=output_path,
    )

    logger.info("Training complete! Results saved to {}", results_dir)
    return results_dir
