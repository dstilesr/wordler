import torch
import rich
import typer
from pathlib import Path
from loguru import logger
from typing import Literal

from wordler.environment import Environment
from wordler.models import ActorModel
from wordler.training.trainer import load_vocabulary

app = typer.Typer(name="Wordle Game")


def display_guess_with_feedback(guess: str, feedback: list[int]) -> None:
    """
    Display a guess with colored letters based on feedback.

    :param guess: The guessed word
    :param feedback: Feedback for each letter (0=not in word, 1=wrong position, 2=correct position)
    """
    # Move cursor up one line, clear it, and return to start
    print("\033[A\033[2K\r", end="")

    # Print colored feedback
    for letter, rsp in zip(guess, feedback):
        if rsp == 0:
            color = "white"
        elif rsp == 1:
            color = "yellow"
        else:
            color = "green"

        rich.print(f"[bold {color}]{letter}[/bold {color}]", end="", sep="")

    print()


@app.command()
def play_game(word: str | None = None):
    """
    Play a wordle game.
    """
    if word:
        env = Environment(word)
    else:
        env = Environment.from_random_word()

    rich.print(
        "Starting wordle game. You have [bold]6 attempts[/bold] to guess a 5 letter word. Begin!"
    )
    while not env.ended:
        guess = input("Next Guess: ")
        guess = guess.strip().lower()
        try:
            fb = env.evaluate_guess(guess)
            display_guess_with_feedback(guess, fb.feedback)

        except ValueError as e:
            rich.print("[bold red]Invalid guess![/bold red] %s" % str(e))
            input("Hit return to continue")
            print("\033[A\033[2K\r", end="")
            print("\033[A\033[2K\r", end="")
            print("\033[A\033[2K\r", end="")

    if env.won:
        rich.print(
            f"Congratulations! You correctly guessed [bold green]{env.word}[/bold green]!"
        )
    else:
        rich.print(
            f"I'm afraid you lost! The correct word was [bold red]{env.word}[/bold red]"
        )


@app.command()
def play_model(
    checkpoint_dir: Path,
    word: str | None = None,
    sample_mode: Literal["deterministic", "probability"] = "deterministic",
):
    """
    Load a model and have it play a game of wordle!
    """
    # Load vocabulary
    vocabulary = load_vocabulary()
    vocabulary_size = len(vocabulary)

    # Load model from checkpoint
    model = ActorModel.from_checkpoint(
        checkpoint_dir=checkpoint_dir,
        vocabulary_size=vocabulary_size,
    )
    model.eval()  # Set to evaluation mode

    # Create environment
    if word:
        env = Environment(word)
    else:
        env = Environment.from_random_word()

    rich.print(
        f"Model loaded from [bold cyan]{checkpoint_dir}[/bold cyan]. "
        f"Watch it play Wordle with [bold]{sample_mode}[/bold] sampling!"
    )
    rich.print("Press [bold]Enter[/bold] to see each guess.\n")

    # Game loop
    while not env.ended:
        # Wait for user to hit enter
        input("Press Enter for next guess...")

        # Get current state
        curr_guesses, curr_feedback = env.get_state()

        # Get model prediction
        with torch.no_grad():
            logits = model(
                torch.tensor([curr_guesses]), torch.tensor([curr_feedback])
            )

            # Select action based on sample mode
            if sample_mode == "deterministic":
                idx = int(torch.argmax(logits).numpy())
            else:  # probability sampling
                idx = int(
                    torch.distributions.Categorical(logits=logits.flatten())
                    .sample((1,))
                    .numpy()[0]
                )

            guess = vocabulary[idx]

        # Evaluate guess
        fb = env.evaluate_guess(guess)
        display_guess_with_feedback(guess, fb.feedback)

    # Game over
    if env.won:
        rich.print(
            f"The model won! It correctly guessed [bold green]{env.word}[/bold green]!"
        )
    else:
        rich.print(
            f"The model lost! The correct word was [bold red]{env.word}[/bold red]"
        )


if __name__ == "__main__":
    import sys

    logger.remove()
    logger.add(sys.stderr, level="ERROR")
    app()
