import torch
import pytest
import numpy as np

from wordler.models.actor_model import ActorModel
from wordler.models.settings import ActorModelSettings


@pytest.fixture(scope="session")
def model_settings() -> ActorModelSettings:
    """
    Fixture to provide a default ActorModelSettings instance.
    """
    return ActorModelSettings(
        embedding_dim=16,
        num_seq_layers=2,
        hidden_dim=16,
        dropout=0.2,
    )


@pytest.fixture(scope="session")
def model(model_settings) -> ActorModel:
    """
    Fixture to provide a default ActorModel instance.
    """
    return ActorModel(
        settings=model_settings,
        vocabulary_size=32,  #: Example vocabulary size
    )


def test_forward_pass(model):
    """
    Test the forward pass of the ActorModel.
    """
    letters_seq = torch.randint(
        low=0,
        high=28,
        size=(1, 5),  #: Batch size of 1 and sequence length of 5
        dtype=torch.long
    )
    feedback_seq = torch.randint(
        low=0,
        high=5,
        size=(1, 5),  #: Batch size of 1 and sequence length of 5
        dtype=torch.long
    )
    mask = torch.ones((1, 5), dtype=torch.bool)

    with torch.no_grad():
        output = model(
            letters_seq,
            feedback_seq,
            mask=mask
        )
        output_2 = model(
            letters_seq,
            feedback_seq,
            mask=None
        )

    assert output.shape == (1, 32), "Output shape is incorrect."
    assert output_2.shape == (1, 32), "Output shape is incorrect."
