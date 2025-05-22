from dataclasses import dataclass, field
from typing import List, Tuple

from .constants import TOKEN_MAP, RESULT_MAP


@dataclass
class GameState:
    """
    Tracks state of the game.
    """
    guessed_words: List[str] = field(default_factory=list)
    guess_results: List[List[int]] = field(default_factory=list)

    def get_input_sequences(self) -> Tuple[List[int], List[int]]:
        """
        Get input sequences for the models.
        :return: Tuple of tokenized words and results.
        """
        return (
            self.words_to_tokens(self.guessed_words),
            self.results_to_tokens(self.guess_results)
        )

    @staticmethod
    def words_to_tokens(words: List[str]) -> List[int]:
        """
        Convert words to sequence of 'tokens'.
        :param words:
        :return:
        """
        out = [TOKEN_MAP["[SEP]"]]
        for word in words:
            out.extend([TOKEN_MAP[letter] for letter in word])
            out.append(TOKEN_MAP["[SEP]"])
        return out

    @staticmethod
    def results_to_tokens(results: List[List[int]]) -> List[int]:
        """
        Convert results to sequence of 'tokens'.
        :param results:
        :return:
        """
        out = [RESULT_MAP["[SEP]"]]
        for result in results:
            out.extend([RESULT_MAP[i] for i in result])
            out.append(RESULT_MAP["[SEP]"])
        return out
