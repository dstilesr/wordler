import pytest

from wordler.environment import Environment, EnvSettings


@pytest.fixture
def env_settings() -> EnvSettings:
    """
    Fixture to provide a default EnvSettings instance.
    """
    return EnvSettings()


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
    assert result_1 == [0, 0, 0, 0, 0], "Incorrectly evaluated guess with no matching letters"

    guess_2 = "block"
    result_2 = environment.evaluate_guess(guess_2)
    assert result_2 == [0, 0, 1, 0, 0], "Incorrectly evaluated guess with one matching letter"

    guess_3 = "shore"
    result_3 = environment.evaluate_guess(guess_3)
    assert result_3 == [1, 1, 1, 1, 2], "Incorrectly evaluated guess with unsorted letters"

    guess_4 = "horse"
    result_4 = environment.evaluate_guess(guess_4)
    assert result_4 == [2, 2, 2, 2, 2], "Incorrectly evaluated correct guess"

