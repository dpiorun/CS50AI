from copy import deepcopy
import pytest

from crossword import *
from generate import *


@pytest.fixture
def crossword0():
    return Crossword("data/structure0.txt", "data/words0.txt")


def test_enforce_node_consistency(crossword0):
    crossword_creator = CrosswordCreator(crossword0)
    expected = {
        Variable(0, 1, 'down', 5): {'SEVEN', 'EIGHT', 'THREE'},
        Variable(1, 4, 'down', 4): {'NINE', 'FOUR', 'FIVE'},
        Variable(0, 1, 'across', 3): {'SIX', 'TEN', 'TWO', 'ONE'},
        Variable(4, 1, 'across', 4): {'NINE', 'FOUR', 'FIVE'}
    }
    crossword_creator.enforce_node_consistency()
    assert crossword_creator.domains == expected


@pytest.mark.parametrize(
    "x,y,expected_x",
    [
        (
            Variable(0, 1, 'across', 3),
            Variable(0, 1, 'down', 5),
            {'TWO', 'THREE', 'FOUR', 'FIVE', 'SEVEN', 'SIX', 'TEN'}
        )
    ]
)
def test_revise(x, y, expected_x, crossword0):
    crossword_creator = CrosswordCreator(crossword0)
    expected_y = deepcopy(crossword_creator.domains[y])

    output = crossword_creator.revise(x, y)

    assert crossword_creator.domains[x] == expected_x
    assert crossword_creator.domains[y] == expected_y
    assert output == True
