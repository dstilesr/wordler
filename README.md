# Wordler

## Description
This is an implementation of a simple bot trained to play the "wordle" game. The bot is trained using some basic
Reinforcement Learning methods to learn how to play the game. Namely, it uses a Monte-Carlo style REINFORCE method
during which, it will iteratively:
- Play a game and track the rewards obtained.
- Compute updates using the Policy Gradient Theorem.

## Scripts

### Training
To train a model, run the `train.py` script. You can run `python train.py --help` to see available options. Training
hyperparameters and configurations are set via environment variables. Check the `.env.example` file to see available
settings.

### Play
You can play the game yourself or see a trained model play with the `play.py` script, with either
`python play.py play-game` or `python play.py play-model`. You can run these with the `--help` flag to see
available options for the commands.

### Development
The project uses `ruff` for formatting / linting and `pytest` for unit tests. You can run unit tests with `uv run pytest`
and run code formatting with `uv run ruff format`.
