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
