from copy import deepcopy
import pytest

from crossword import *
from generate import *


@pytest.fixture
def crossword0_creator():
    crossword = Crossword("data/structure0.txt", "data/words0.txt")
    return CrosswordCreator(crossword)


@pytest.fixture
def crossword1_creator():
    crossword = Crossword("data/structure1.txt", "data/words1.txt")
    return CrosswordCreator(crossword)


@pytest.fixture
def failed_crossword_creator():
    crossword = Crossword("data/structure0.txt",
                          "data/words0_with_one_missing.txt")
    return CrosswordCreator(crossword)


def test_enforce_node_consistency(crossword0_creator: CrosswordCreator):
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


@pytest.fixture
def crossword0_creator_node_consistent(crossword0_creator: CrosswordCreator):
    crossword0_creator.enforce_node_consistency()
    return crossword0_creator


@pytest.fixture
def crossword1_creator_node_consistent(crossword1_creator: CrosswordCreator):
    crossword1_creator.enforce_node_consistency()
    return crossword1_creator


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
def test_revise(x: Variable, y: Variable, expected_x: set[str], expected_output: bool, crossword0_creator: CrosswordCreator):
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


def test_ac3(crossword0_creator_node_consistent: CrosswordCreator):
    """
    It should begin with initial list of all arcs in the problem, if no arcs are provided.
    """
    expected = {
        Variable(0, 1, 'down', 5): {'SEVEN'},
        Variable(0, 1, 'across', 3): {'SIX'},
        Variable(1, 4, 'down', 4): {'FIVE'},
        Variable(4, 1, 'across', 4): {'NINE'}
    }
    output = crossword0_creator_node_consistent.ac3()

    assert crossword0_creator_node_consistent.domains == expected
    assert output == True


def test_ac3_with_arcs(crossword0_creator_node_consistent: CrosswordCreator):
    """
    It should not change domains if empty list of arcs was provided.
    """
    arcs = list()
    expected = deepcopy(crossword0_creator_node_consistent.domains)

    output = crossword0_creator_node_consistent.ac3(arcs)

    assert crossword0_creator_node_consistent.domains == expected
    assert output == True


def test_ac3_with_no_solution(failed_crossword_creator: CrosswordCreator):
    """
    It should return false if there is no possible solution.
    """
    failed_crossword_creator.enforce_node_consistency()
    output = failed_crossword_creator.ac3()

    assert output == False


@pytest.mark.parametrize(
    "assignment,expected",
    [
        (
            {
                Variable(0, 1, 'down', 5): "foo",
                Variable(1, 4, 'down', 4): "foo",
                Variable(0, 1, 'across', 3): "foo",
                Variable(4, 1, 'across', 4): "foo"
            },
            True
        ),
        (
            {
                Variable(0, 1, 'down', 5): "foo"
            },
            False
        )
    ]
)
def test_assignment_complete(assignment: dict, expected: bool, crossword0_creator: CrosswordCreator):
    """
    It should return True if `assignment` is complete (i.e., assigns a value to each
    crossword variable); return False otherwise.    
    """
    output = crossword0_creator.assignment_complete(assignment)
    assert output == expected


@pytest.mark.parametrize(
    "assignment,expected",
    [
        (
            {
                Variable(0, 1, 'down', 5): 'SEVEN',
                Variable(0, 1, 'across', 3): 'ONE',
            },
            False
        ),
        (
            {
                Variable(1, 4, 'down', 4): 'FIVE',
                Variable(4, 1, 'across', 4): 'FIVE'
            },
            False
        ),
        (
            {
                Variable(0, 1, 'down', 5): 'SEVEN',
                Variable(1, 4, 'down', 4): 'FIVE',
                Variable(0, 1, 'across', 3): 'ONE',
                Variable(4, 1, 'across', 4): 'NINE'
            },
            False
        ),
        (
            {
                Variable(0, 1, 'down', 5): 'SEVEN',
                Variable(0, 1, 'across', 3): 'SIX',
            },
            True
        ),
        (
            {
                Variable(0, 1, 'down', 5): 'SEVEN',
                Variable(0, 1, 'across', 3): 'SIX',
                Variable(1, 4, 'down', 4): 'FIVE',
                Variable(4, 1, 'across', 4): 'NINE'
            },
            True
        ),
        (
            {
                Variable(0, 1, 'down', 5): 'SEVEN',
            },
            True
        ),
        (
            {},
            True
        ),
    ]
)
def test_consistent(assignment: dict, expected: bool, crossword0_creator: CrosswordCreator):
    """
    * An assignment is a dictionary where the keys are Variable objects and the values are strings
    representing the words those variables will take on. Note that the assignment may not be complete:
    not all variables will necessarily be present in the assignment.
    * An assignment is consistent if it satisfies all of the constraints of the problem: that is to say,
    all values are distinct, every value is the correct length,
    and there are no conflicts between neighboring variables.
    * The function should return True if the assignment is consistent and return False otherwise
    """
    output = crossword0_creator.consistent(assignment)
    assert output == expected


@pytest.mark.parametrize(
    "var,assignment,expected",
    [
        (
            Variable(4, 1, 'across', 4),
            {},
            # eliminates accordingly `n` choices [4, 5, 6]
            ['NINE', 'FIVE', 'FOUR']
        ),
        (
            Variable(4, 1, 'across', 4),
            {Variable(1, 4, 'down', 4): 'FIVE'},
            # eliminates accordingly `n` choices [4, 5]
            ['NINE', 'FOUR']
        )
    ]

)
def test_order_domain_values(var: Variable, assignment: dict, expected: list[str], crossword0_creator_node_consistent: CrosswordCreator):
    """
    It should return a list of all of the values in the domain of var, ordered according to the least-constraining values heuristic.
    * `var` will be a Variable object, representing a variable in the puzzle.
    * Recall that the least-constraining values heuristic is computed as the number of values
      ruled out for neighboring unassigned variables. That is to say, if assigning var to a particular
      value results in eliminating `n` possible choices for neighboring variables, it should order its
      results in ascending order of `n`.
    * Note that any variable present in `assignment already` has a value, and therefore shouldn’t
      be counted when computing the number of values ruled out for neighboring unassigned variables.
    * For domain values that eliminate the same number of possible choices for neighboring variables,
      any ordering is acceptable.

    self.domains = {
        Variable(0, 1, 'across', 3): {'TWO', 'ONE', 'SIX', 'TEN'}, 
        Variable(0, 1, 'down', 5): {'THREE', 'EIGHT', 'SEVEN'}, 
        Variable(4, 1, 'across', 4): {'NINE', 'FOUR', 'FIVE'}, 
        Variable(1, 4, 'down', 4): {'NINE', 'FOUR', 'FIVE'}
    }
    """
    output = crossword0_creator_node_consistent.order_domain_values(
        var, assignment)
    assert output == expected


@pytest.mark.parametrize(
    "assignment,expected_in",
    [
        (
            {},
            [
                Variable(0, 1, 'down', 5),
                Variable(4, 1, 'across', 4),
            ]
        ),
        (
            {
                Variable(0, 1, 'across', 3): 'SIX'
            },
            [
                Variable(4, 1, 'across', 4),
            ]
        ),
        (
            {
                Variable(0, 1, 'down', 5): 'SEVEN',
                Variable(4, 1, 'across', 4): 'NINE',
                Variable(1, 4, 'down', 4): 'FIVE'
            },
            [
                Variable(0, 1, 'across', 3),
            ]
        )
    ]
)
def test_select_unassigned_variable(assignment: dict[Variable: str], expected_in: list[Variable], crossword0_creator_node_consistent: CrosswordCreator):
    """
    It should return a single variable in the crossword puzzle that is not yet assigned by assignment,
    according to the minimum remaining value heuristic and then the degree heuristic.
    * An assignment is a dictionary where the keys are Variable objects and the values are strings
      representing the words those variables will take on. It is assumed that the assignment
      will not be complete: not all variables will be present in the assignment.
    * It should return a Variable object. It should return the variable with the fewest number
      of remaining values in its domain. If there is a tie between variables, it should choose
      among whichever among those variables has the largest degree (has the most neighbors).
      If there is a tie in both cases, it may choose arbitrarily among tied variables.

    self.domains = {
        Variable(0, 1, 'across', 3): {'TWO', 'ONE', 'SIX', 'TEN'}, 
        Variable(0, 1, 'down', 5): {'THREE', 'EIGHT', 'SEVEN'}, 
        Variable(4, 1, 'across', 4): {'NINE', 'FOUR', 'FIVE'}, 
        Variable(1, 4, 'down', 4): {'NINE', 'FOUR', 'FIVE'}
    }
    """
    output = crossword0_creator_node_consistent.select_unassigned_variable(
        assignment
    )
    assert output in expected_in


def test_backtrack(crossword1_creator_node_consistent: CrosswordCreator):
    """
    It should return a complete satisfactory assignment of variables
    to values if it is possible to do so.
    """
    crossword1_creator_node_consistent.ac3()
    expected_in_domains = {
        Variable(2, 1, 'across', 12): {'INTELLIGENCE'},
        Variable(4, 4, 'across', 5): {'LOGIC'},
        Variable(6, 5, 'across', 6): {'SEARCH', 'REASON'},
        Variable(2, 1, 'down', 5): {'INFER'},
        Variable(1, 7, 'down', 7): {'MINIMAX'},
        Variable(1, 12, 'down', 7): {'RESOLVE', 'NETWORK'}
    }
    output = crossword1_creator_node_consistent.backtrack({})
    for var in expected_in_domains:
        assert output[var] in expected_in_domains[var]


def test_backtrack_return_None(crossword1_creator_node_consistent: CrosswordCreator):
    """
    It should return None if it not possible to find complete satisfactory assignment.
    """
    crossword1_creator_node_consistent.ac3()
    output = crossword1_creator_node_consistent.backtrack({
        Variable(2, 1, 'across', 12): "bad_word"
    })
    assert output == None
