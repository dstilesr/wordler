import sys
from loguru import logger
from pathlib import Path
from typing import Literal

from wordler.training.run_train import run_training


def main(
    total_episodes: int = 60_000,
    output_path: Path | None = None,
    model_path: Path | None = None,
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO",
):
    """
    Run training for the wordle model.
    :param total_episodes: Total episodes to run for training.
    :param output_path: Path to store outputs. 'output' directory in repo root by default.
    :param model_path: Optional path to existing model to continue training.
    :param log_level:
    :return:
    """
    logger.remove()
    logger.add(sys.stderr, level=log_level)

    output_path = output_path or Path(__file__).parent / "outputs"
    if not output_path.is_dir():
        output_path.mkdir(parents=True)

    run_training(total_episodes, output_path, model_path)


if __name__ == "__main__":
    import typer

    typer.run(main)
