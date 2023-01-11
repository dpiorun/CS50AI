from typing import Union
import pytest

from nim import *


@pytest.fixture
def nimAI():
    return NimAI()


@pytest.mark.parametrize(
    "state,action,expected",
    [
        (
            (1, 3, 5, 7),
            (1, 1),
            0
        ),
        (
            [1, 3, 5, 7],
            (1, 1),
            0
        )
    ]
)
def test_get_q_value(state: Union[tuple[int], list[int]], action: tuple[int], expected: int, nimAI: NimAI):
    """
    It should return 0 if no Q-value exists in self.q
    """
    output = nimAI.get_q_value(state, action)
    assert output == expected
