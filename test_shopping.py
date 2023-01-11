import pytest
from typing import Union

from shopping import *


@pytest.mark.parametrize(
    "index,expected_evidence,expected_label",
    [
        (
            0,
            [
                0, 0.0, 0, 0.0, 1, 0.0, 0.2, 0.2, 0.0, 0.0, 1, 1, 1, 1, 1, 1, 0
            ],
            0
        ),
        (
            76,
            [
                10, 1005.666667, 0, 0, 36, 2111.341667, 0.004347826, 0.014492754, 11.43941195, 0, 1, 2, 6, 1, 2, 1, 0
            ],
            1
        )

    ]
)
def test_load_data(index: int, expected_evidence: list[Union[int, float]], expected_label: int):
    """
    It should load row as a tuple.
    """
    evidence, labels = load_data('shopping.csv')
    NUM_OF_ROWS = 12330
    assert evidence[index] == expected_evidence
    assert labels[index] == expected_label
    assert len(evidence) == len(labels) == NUM_OF_ROWS


@pytest.mark.parametrize(
    "labels,predictions,expected_sensitivity,expected_specificity",
    [
        (
            [1, 0],
            [1, 0],
            1,
            1
        ),
        (
            [1, 0],
            [0, 1],
            0,
            0
        ),
        (
            [1, 0],
            [1, 1],
            1,
            0
        ),
        (
            [1, 0],
            [0, 0],
            0,
            1
        ),
        (
            [1, 1, 0, 0],
            [1, 0, 0, 1],
            0.5,
            0.5
        )
    ]
)
def test_evaluate(labels: list[int], predictions: list[int], expected_sensitivity: float, expected_specificity: float):
    sensitivity, specificity = evaluate(labels, predictions)
    assert sensitivity == expected_sensitivity
    assert specificity == expected_specificity
