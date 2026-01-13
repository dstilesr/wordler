import rich
import typer
from loguru import logger

from wordler.environment import Environment

app = typer.Typer(name="Wordle Game")


@app.command()
def play_game(word: str | None = None):
    """
    Play a wordle game.
    """
    if word:
        env = Environment(word)
    else :
        env = Environment.from_random_word()

    rich.print("Starting wordle game. You have [bold]6 attempts[/bold] to guess a 5 letter word. Begin!")
    while not env.ended:
        guess = input("Next Guess: ")
        guess = guess.strip().lower()
        try:
            fb = env.evaluate_guess(guess)

            # Move cursor up one line, clear it, and return to start
            print("\033[A\033[2K\r", end="")

            # Print colored feedback
            for letter, rsp in zip(guess, fb.feedback):
                if rsp == 0:
                    color = "white"
                elif rsp == 1:
                    color = "yellow"
                else:
                    color = "green"

                rich.print(f"[bold {color}]{letter}[/bold {color}]", end="", sep="")

            print()

        except ValueError as e:
            rich.print("[bold red]Invalid guess![/bold red] %s" % str(e))
            input("Hit return to continue")
            print("\033[A\033[2K\r", end="")
            print("\033[A\033[2K\r", end="")
            print("\033[A\033[2K\r", end="")


    if env.won:
        rich.print(f"Congratulations! You correctly guessed [bold green]{env.word}[/bold green]!")
    else:
        rich.print(f"I'm afraid you lost! The correct word was [bold red]{env.word}[/bold red]")


if __name__ == '__main__':
    import sys

    logger.remove()
    logger.add(sys.stderr, level="ERROR")
    app()
