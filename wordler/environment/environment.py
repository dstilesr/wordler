import random
from loguru import logger
from typing import Optional, Set, List

from .settings import EnvSettings
from ..constants import WORDS_FILE, TOKEN_MAP, RESULT_MAP


class Environment:
    """
    Wordle game environment.
    """

    @staticmethod
    def load_word_set() -> Set[str]:
        """
        Load the word set from the word list file.
        :return:
        """
        with WORDS_FILE.open("r") as f:
            words = f.read().strip().split("\n")
        return set(words)

    def __init__(
        self,
        word: str,
        settings: Optional[EnvSettings] = None,
        words_set: Optional[Set[str]] = None,
    ):
        """
        Initialize the environment with a word.
        :param word:
        :param settings:
        :param words_set:
        """
        self.settings = settings or EnvSettings()
        self.words_set = words_set or self.load_word_set()
        if not self.validate_word(word):
            raise ValueError(f"Invalid word: {word}")

        self.__word = word
        self.__guesses = []
        self.__guesses_vec = [TOKEN_MAP["[CLS]"]]
        self.__feedback_vec = [RESULT_MAP["[CLS]"]]

    @property
    def word(self) -> str:
        """
        The target word to guess.
        """
        return self.__word

    @property
    def guesses(self) -> list[list[str]]:
        """
        List of guesses that have been submitted by the player.
        """
        return self.__guesses

    def validate_word(self, word: str) -> bool:
        """
        Check that the word is valid for wordle.
        :param word:
        :return:
        """
        return (
            (len(word) == 5)
            and word.isalpha()
            and word.islower()
            and (word in self.words_set)
        )

    @classmethod
    def from_random_word(
        cls,
        settings: Optional[EnvSettings] = None,
        random_seed: Optional[int] = None,
    ) -> "Environment":
        """
        Create a new environment with a random word.
        :param settings:
        :param random_seed:
        :return:
        """
        words = cls.load_word_set()
        if random_seed is not None:
            random.seed(random_seed)

        lst = list(words)
        lst.sort()
        word = random.choice(lst)
        return cls(word, settings=settings, words_set=words)

    def __update_state(self, guess: str, feedback: list[int]):
        """
        Update the internal state with the guess and the feedback received for it.
        :param guess:
        :param feedback:
        :return:
        """
        self.__guesses.append(guess)

        self.__guesses_vec.extend(TOKEN_MAP[letter] for letter in guess)
        self.__guesses_vec.append(TOKEN_MAP["[SEP]"])

        self.__feedback_vec.extend(RESULT_MAP[fb] for fb in feedback)
        self.__feedback_vec.append(RESULT_MAP["[SEP]"])

    def get_state(self) -> tuple[List[int], List[int]]:
        """
        Get the tuple of guess vector, feedback vector that represent the state
        of the game.
        :return:
        """
        return self.__guesses_vec, self.__feedback_vec

    def evaluate_guess(self, guess: str) -> List[int]:
        """
        Evaluate a guess. For each letter in the guess, return:
        - 2 if the letter is in the correct position
        - 1 if the letter is in the word but in the wrong position
        - 0 if the letter is not in the word

        Updates the list of guesses that have been submitted for the given game
        and the guess and feedback vectors as well.
        :param guess:
        :return:
        """
        if not self.validate_word(guess):
            raise ValueError(f"Invalid guess: {guess}")
        elif len(self.guesses) > 5:
            raise ValueError("Maximum number of guesses submitted already!")

        logger.debug("Guess submitted: '{}'", guess)
        values = [0] * 5

        unmatched = []
        for idx, (guess_letter, real) in enumerate(zip(guess, self.word)):
            if guess_letter == real:
                values[idx] = 2
                continue

            unmatched.append(idx)

        logger.debug(
            "Unmatched letter indices: {}", ", ".join(map(str, unmatched))
        )

        # Evaluate unmatched letters
        remaining = [self.word[idx] for idx in unmatched]
        remaining_guess = [guess[idx] for idx in unmatched]
        for idx, letter in zip(unmatched, remaining_guess):
            if letter in remaining:
                values[idx] = 1
                remaining.remove(letter)

        self.__update_state(guess, values)
        return values
