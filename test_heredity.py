import pytest
from heredity import *


@pytest.mark.parametrize(
    "people",
    [
        {
            "Harry": {
                "name": "Harry",
                "mother": "Lily",
                "father": "James",
                "trait": None,
            },
            "James": {"name": "James", "mother": None, "father": None, "trait": True},
            "Lily": {"name": "Lily", "mother": None, "father": None, "trait": False},
        }
    ],
)
@pytest.mark.parametrize(
    "one_gene,two_genes,have_trait,expected",
    [
        (
            {"Harry"},
            {"James"},
            {"James"},
            0.0026643247488,
        ),
        (
            {"Harry"},
            {"James"},
            {"James", "Harry"},
            0.0033909587712,
        ),
    ],
)
def test_joint_probability(people, one_gene, two_genes, have_trait, expected):
    output = joint_probability(people, one_gene, two_genes, have_trait)
    assert output == expected
