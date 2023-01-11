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


@pytest.mark.parametrize(
    "state,action,old_q,reward,future_rewards,expected",
    [
        (
            (1, 3, 5, 7),
            (1, 1),
            0,
            1,
            1,
            {
                ((1, 3, 5, 7), (1, 1)): 1
            }
        ),
        (
            (1, 3, 5, 7),
            (1, 1),
            1,
            1,
            1,
            {
                ((1, 3, 5, 7), (1, 1)): 1.5
            }
        )
    ]

)
def test_update_q_value(
    state: Union[tuple[int], list[int]],
    action: tuple[int],
    old_q: float,
    reward: int,
    future_rewards: int,
    expected: dict[tuple[tuple[int]]: int],
    nimAI: NimAI
):
    """
    It should update q value.
    """
    nimAI.update_q_value(state, action, old_q, reward, future_rewards)
    assert nimAI.q == expected


@pytest.mark.parametrize(
    "state",
    [
        (1, 3, 5, 7),
        (0, 0, 0, 0)
    ]
)
def test_best_future_reward(state: Union[list, tuple], nimAI: NimAI):
    """
    It should return 0 if q is empty or there is not possible action.
    """
    output = nimAI.best_future_reward(state)
    assert output == 0
