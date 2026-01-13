import torch
import random
from tqdm import tqdm
from loguru import logger
from dataclasses import dataclass, field

from ..models import ActorModel
from .settings import TrainSettings
from ..environment import Environment, EnvSettings, GuessResult


def load_vocabulary() -> list[str]:
    """
    Load the vocabulary.
    """
    vocab = list(Environment.load_word_set())
    vocab.sort()
    logger.info("Loaded vocabulary with {} entries", len(vocab))
    return vocab


@dataclass()
class Trainer:
    """
    Represents trainer that plays games and updates state as it goes.
    """

    model: ActorModel
    game_settings: EnvSettings
    current_game: Environment | None = field(default=None)
    settings: TrainSettings = field(default_factory=TrainSettings)
    vocabulary: list[str] = field(default_factory=load_vocabulary)
    total_rewards: list[float] = field(default_factory=list)
    wins: list[bool] = field(default_factory=list)

    def perform_update(
        self,
        logits: list[torch.Tensor],
        rewards: list[float],
        actions: list[int],
        optimizer: torch.optim.Optimizer,
    ):
        """
        Perform gradient updates using an episode trajectory.
        :param logits:
        :param rewards:
        :param actions:
        :param optimizer:
        :return:
        """
        steps = len(rewards)
        returns = [0.0 for _ in range(steps)]
        returns[-1] = rewards[-1]
        for i in range(1, steps):
            returns[steps - 1 - i] = (
                rewards[steps - 1 - i]
                + self.settings.discount * returns[steps - i]
            )

        optimizer.zero_grad()
        loss = 0.0
        for i in range(steps):
            dist = torch.distributions.Categorical(logits=logits[i].flatten())
            logprob = dist.log_prob(torch.tensor(actions[i]))
            loss -= returns[i] * logprob

        loss.backward()
        optimizer.step()

    def run_game(
        self,
        optimizer: torch.optim.Optimizer,
        target: str | None = None,
        temperature: float = 1.0,
    ):
        """
        Run a game and update the model's values as it plays with the given
        rewards.
        """
        self.current_game = (
            Environment(word=target, settings=self.game_settings)
            if target
            else Environment.from_random_word(self.game_settings)
        )
        game_rewards = []
        game_states = [self.current_game.get_state()]
        actions = []
        game_logits = []

        # Generate episode trajectory
        while not self.current_game.ended:
            curr_guesses, curr_feedback = game_states[-1]
            logits = self.model(
                torch.tensor([curr_guesses]), torch.tensor([curr_feedback])
            )
            with torch.no_grad():
                # Select action
                idx = int(
                    torch.distributions.Categorical(
                        logits=logits.flatten() / temperature
                    )
                    .sample((1,))
                    .numpy()[0]
                )
                guess = self.vocabulary[idx]

            result = self.current_game.evaluate_guess(guess)

            actions.append(idx)
            game_rewards.append(result.reward)
            game_states.append(self.current_game.get_state())
            game_logits.append(logits)

        self.perform_update(game_logits, game_rewards, actions, optimizer)
        self.wins.append(self.current_game.won)
        self.total_rewards.append(sum(game_rewards))

    def train(self, total_games: int = 60_000):
        """
        Run wordle games to train the agent.
        :param total_games:
        :return:
        """
        self.model.train()
        coef = (1 - self.settings.temperature) / total_games
        optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self.settings.learning_rate
        )
        for i in tqdm(range(total_games)):
            temperature = coef * i + self.settings.temperature
            self.run_game(
                optimizer, random.choice(self.vocabulary), temperature
            )
