import torch
import pytest
import numpy as np

from wordler.models.self_attn import SelfAttention


@pytest.fixture(scope="function")
def sequence() -> torch.Tensor:
    """
    Fixture to provide a fixed sequence for the self-attention layer.
    Shape: (1, 2, 4)
    """
    out = torch.tensor([[
        [1, 1, 1, 1],
        [2, 2, 2, 2],
    ]])
    return out.to(dtype=torch.float32)


@pytest.fixture(scope="function")
def attention_layer() -> SelfAttention:
    """
    Fixture to provide a self-attention layer for testing.
    """
    return SelfAttention(
        input_dim=4,
        output_dim=6,
        dtype=torch.float32,
    )


def test_tensor_shape(sequence, attention_layer):
    """
    Test the shape of the output tensor.
    :param sequence:
    :param attention_layer:
    :return:
    """
    with torch.no_grad():
        output = attention_layer(
            x=sequence,
        )
    assert output.shape == (1, 2, 6), "Output tensor shape is incorrect."

