from copy import deepcopy
import pytest

from crossword import *
from generate import *


@pytest.fixture
def crossword0_creator():
    crossword = Crossword("data/structure0.txt", "data/words0.txt")
    return CrosswordCreator(crossword)


@pytest.fixture
def failed_crossword_creator():
    crossword = Crossword("data/structure0.txt",
                          "data/words0_with_one_missing.txt")
    return CrosswordCreator(crossword)


def test_enforce_node_consistency(crossword0_creator):
    """
    It should remove any values that are inconsistent with a variable's unary
    constraints; in this case, the length of the word.
    """
    expected = {
        Variable(0, 1, 'down', 5): {'SEVEN', 'EIGHT', 'THREE'},
        Variable(1, 4, 'down', 4): {'NINE', 'FOUR', 'FIVE'},
        Variable(0, 1, 'across', 3): {'SIX', 'TEN', 'TWO', 'ONE'},
        Variable(4, 1, 'across', 4): {'NINE', 'FOUR', 'FIVE'}
    }
    crossword0_creator.enforce_node_consistency()
    assert crossword0_creator.domains == expected


@pytest.mark.parametrize(
    "x,y,expected_x,expected_output",
    [
        (
            Variable(0, 1, 'across', 3),
            Variable(0, 1, 'down', 5),
            {'TWO', 'THREE', 'FOUR', 'FIVE', 'SEVEN', 'SIX', 'TEN'},
            True
        ),
        (
            Variable(0, 1, 'across', 3),
            Variable(4, 1, 'across', 4),
            {'ONE', 'TWO', 'THREE', 'FOUR', 'FIVE',
                'SIX', 'SEVEN', 'EIGHT', 'NINE', 'TEN'},  # x and y have no overlap. Every value corresponds.
            False
        )
    ]
)
def test_revise(x, y, expected_x, expected_output, crossword0_creator):
    """
    It should remove values from `domains[x]` for which there is no
    possible corresponding value for `y` in `domains[y]`.
    """
    expected_y = deepcopy(
        crossword0_creator.domains[y])  # it should not change Y domain

    output = crossword0_creator.revise(x, y)

    assert crossword0_creator.domains[x] == expected_x
    assert crossword0_creator.domains[y] == expected_y
    assert output == expected_output


def test_ac3(crossword0_creator):
    """
    It should begin with initial list of all arcs in the problem, if no arcs are provided.
    """
    crossword0_creator.enforce_node_consistency()
    expected = {
        Variable(0, 1, 'down', 5): {'SEVEN'},
        Variable(0, 1, 'across', 3): {'SIX'},
        Variable(1, 4, 'down', 4): {'FIVE'},
        Variable(4, 1, 'across', 4): {'NINE'}
    }
    output = crossword0_creator.ac3()

    assert crossword0_creator.domains == expected
    assert output == True


def test_ac3_with_arcs(crossword0_creator):
    """
    It should not change domains if empty list of arcs was provided.
    """
    crossword0_creator.enforce_node_consistency()
    arcs = list()
    expected = deepcopy(crossword0_creator.domains)

    output = crossword0_creator.ac3(arcs)

    assert crossword0_creator.domains == expected
    assert output == True


def test_ac3_with_no_solution(failed_crossword_creator):
    """
    It should return false if there is no possible solution.
    """
    failed_crossword_creator.enforce_node_consistency()
    output = failed_crossword_creator.ac3()

    assert output == False
