import sys
import pytest
import numpy as np
from loguru import logger

from wordler import constants as const
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
    result_1 = environment.evaluate_guess(guess_1).feedback
    assert result_1 == [0, 0, 0, 0, 0], (
        "Incorrectly evaluated guess with no matching letters"
    )

    guess_2 = "block"
    result_2 = environment.evaluate_guess(guess_2).feedback
    assert result_2 == [0, 0, 1, 0, 0], (
        "Incorrectly evaluated guess with one matching letter"
    )

    guess_3 = "shore"
    result_3 = environment.evaluate_guess(guess_3).feedback
    assert result_3 == [1, 1, 1, 1, 2], (
        "Incorrectly evaluated guess with unsorted letters"
    )

    guess_4 = "horse"
    result_4 = environment.evaluate_guess(guess_4).feedback
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
    result_1 = environment.evaluate_guess(guess_1).feedback
    assert result_1 == [2, 0, 0, 0, 1], (
        "Incorrectly evaluated guess with repeated letters"
    )

    guess_2 = "feast"
    result_2 = environment.evaluate_guess(guess_2).feedback
    assert result_2 == [0, 1, 0, 1, 2], (
        "Incorrectly evaluated guess with repeated letters"
    )

    guess_3 = "worse"
    result_3 = environment.evaluate_guess(guess_3).feedback
    assert result_3 == [1, 0, 0, 1, 1], (
        "Incorrectly evaluated guess with repeated letters"
    )

    guess_4 = "reset"
    result_4 = environment.evaluate_guess(guess_4).feedback
    assert result_4 == [0, 1, 1, 2, 2], (
        "Incorrectly evaluated correct guess with repeated letters"
    )

    guess_5 = "sweet"
    result_5 = environment.evaluate_guess(guess_5).feedback
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


def test_state_updates(env_settings):
    """
    Test that the game state is evaluated correctly as guesses are submitted.
    :param env_settings:
    """
    env = Environment(word="purse", settings=env_settings)
    guesses = ["boils", "scold", "curse", "purse"]
    gs, fb = env.get_state()
    assert len(gs) == len(fb) == 1, "Invalid initialization"
    assert (
        gs[0] == const.TOKEN_MAP["[CLS]"] and fb[0] == const.RESULT_MAP["[CLS]"]
    )

    for g in guesses:
        res = env.evaluate_guess(g)
        feedback = res.feedback
        gs, fb = env.get_state()
        assert len(gs) == len(fb), "Sequence length mismatch"

        assert gs[-1] == const.TOKEN_MAP["[SEP]"]
        assert fb[-1] == const.RESULT_MAP["[SEP]"]

        for i in range(len(g)):
            assert gs[-(len(g) + 1 - i)] == const.TOKEN_MAP[g[i]]
            assert fb[-(len(feedback) + 1 - i)] == const.RESULT_MAP[feedback[i]]


def test_reward_computation(env_settings):
    """
    Test that the reward for a guess is computed correctly.
    :param env_settings:
    :return:
    """
    env = Environment(word="horse", settings=env_settings)

    res = env.evaluate_guess("canny")
    assert np.isclose(res.reward, 0.0)

    res = env.evaluate_guess("spunk")
    assert np.isclose(res.reward, env_settings.letter_present_reward)

    res = env.evaluate_guess("carry")
    assert np.isclose(res.reward, env_settings.correct_letter_reward)

    res = env.evaluate_guess("curse")
    assert np.isclose(res.reward, 3 * env_settings.correct_letter_reward)

    res = env.evaluate_guess("horse")
    assert np.isclose(res.reward, env_settings.win_reward)
