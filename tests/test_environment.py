import sys
import pytest
from loguru import logger

from wordler.environment import Environment, EnvSettings


@pytest.fixture
def env_settings() -> EnvSettings:
    """
    Fixture to provide a default EnvSettings instance.
    """
    return EnvSettings()


@pytest.fixture(scope="session", autouse=True)
def logger_setup():
    """
    Set the log level for the session.
    :return:
    """
    logger.remove()
    logger.add(sys.stderr, level="DEBUG", colorize=True)


def test_evaluate_guess(env_settings):
    """
    Test that a valid guess can be correctly evaluated by the environment.
    :param env_settings:
    :return:
    """
    environment = Environment(
        word="horse",
        settings=env_settings,
    )

    guess_1 = "milky"
    result_1 = environment.evaluate_guess(guess_1)
    assert result_1 == [0, 0, 0, 0, 0], (
        "Incorrectly evaluated guess with no matching letters"
    )

    guess_2 = "block"
    result_2 = environment.evaluate_guess(guess_2)
    assert result_2 == [0, 0, 1, 0, 0], (
        "Incorrectly evaluated guess with one matching letter"
    )

    guess_3 = "shore"
    result_3 = environment.evaluate_guess(guess_3)
    assert result_3 == [1, 1, 1, 1, 2], (
        "Incorrectly evaluated guess with unsorted letters"
    )

    guess_4 = "horse"
    result_4 = environment.evaluate_guess(guess_4)
    assert result_4 == [2, 2, 2, 2, 2], "Incorrectly evaluated correct guess"


def test_guess_repeated_letters(env_settings):
    """
    Test guesses when the target word has repeated letters.
    :param env_settings:
    :return:
    """
    environment = Environment(
        word="sweet",
        settings=env_settings,
    )

    guess_1 = "shape"
    result_1 = environment.evaluate_guess(guess_1)
    assert result_1 == [2, 0, 0, 0, 1], (
        "Incorrectly evaluated guess with repeated letters"
    )

    guess_2 = "feast"
    result_2 = environment.evaluate_guess(guess_2)
    assert result_2 == [0, 1, 0, 1, 2], (
        "Incorrectly evaluated guess with repeated letters"
    )

    guess_3 = "worse"
    result_3 = environment.evaluate_guess(guess_3)
    assert result_3 == [1, 0, 0, 1, 1], (
        "Incorrectly evaluated guess with repeated letters"
    )

    guess_4 = "reset"
    result_4 = environment.evaluate_guess(guess_4)
    assert result_4 == [0, 1, 1, 2, 2], (
        "Incorrectly evaluated correct guess with repeated letters"
    )

    guess_5 = "sweet"
    result_5 = environment.evaluate_guess(guess_5)
    assert result_5 == [2, 2, 2, 2, 2], (
        "Incorrectly evaluated correct guess with repeated letters"
    )


def test_invalid_guess(env_settings):
    """
    Test that an invalid guess raises a ValueError.
    :param env_settings:
    :return:
    """
    environment = Environment(
        word="horse",
        settings=env_settings,
    )

    invalid_guesses = [
        "horsey",  # too long
        "hor",  # too short
        "h0rse",  # contains a number
        "hhkhh",  # Not a word
    ]

    for guess in invalid_guesses:
        with pytest.raises(ValueError):
            environment.evaluate_guess(guess)
