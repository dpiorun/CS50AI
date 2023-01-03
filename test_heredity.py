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


@pytest.mark.parametrize(
    "one_gene,two_genes,have_trait,p,expected",
    [
        (
            {"James"},
            {"James"},
            {"Harry"},
            0.33,
            {"Harry": {"gene": {2: 0, 1: 0, 0: 0.33}, "trait": {True: 0.33, False: 0}}},
        ),
        (
            {"James"},
            {"Harry"},
            {"Harry"},
            0.33,
            {"Harry": {"gene": {2: 0.33, 1: 0, 0: 0}, "trait": {True: 0.33, False: 0}}},
        ),
        (
            {"Harry"},
            {"James"},
            {"James"},
            0.33,
            {"Harry": {"gene": {2: 0, 1: 0.33, 0: 0}, "trait": {True: 0, False: 0.33}}},
        ),
    ],
)
def test_update(
    one_gene: set[str],
    two_genes: set[str],
    have_trait: set[str],
    p: float,
    expected: dict,
):
    probabilities = {
        "Harry": {"gene": {2: 0, 1: 0, 0: 0}, "trait": {True: 0, False: 0}}
    }
    update(probabilities, one_gene, two_genes, have_trait, p)
    assert probabilities == expected


@pytest.mark.parametrize(
    "probabilities,expected",
    [
        (
            {
                "Harry": {
                    "gene": {2: 0.1, 1: 0.15, 0: 0.25},
                    "trait": {True: 0.1, False: 0.3},
                }
            },
            {
                "Harry": {
                    "gene": {2: 0.2, 1: 0.3, 0: 0.5},
                    "trait": {True: 0.25, False: 0.75},
                }
            },
        ),
        (
            {
                "Harry": {
                    "gene": {2: 0.2, 1: 0.3, 0: 0.5},
                    "trait": {True: 0.3, False: 0.1},
                }
            },
            {
                "Harry": {
                    "gene": {2: 0.2, 1: 0.3, 0: 0.5},
                    "trait": {True: 0.75, False: 0.25},
                }
            },
        ),
    ],
)
def test_normalize(probabilities: dict, expected: dict):
    normalize(probabilities)
    assert probabilities == expected


def test_main(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["heredity.py", "data/family0.csv"])
    main()
    out, err = capsys.readouterr()
    expected = (
        "Harry:\n"
        + "  Gene:\n"
        + "    2: 0.0092\n"
        + "    1: 0.4557\n"
        + "    0: 0.5351\n"
        + "  Trait:\n"
        + "    True: 0.2665\n"
        + "    False: 0.7335\n"
        + "James:\n"
        + "  Gene:\n"
        + "    2: 0.1976\n"
        + "    1: 0.5106\n"
        + "    0: 0.2918\n"
        + "  Trait:\n"
        + "    True: 1.0000\n"
        + "    False: 0.0000\n"
        + "Lily:\n"
        + "  Gene:\n"
        + "    2: 0.0036\n"
        + "    1: 0.0136\n"
        + "    0: 0.9827\n"
        + "  Trait:\n"
        + "    True: 0.0000\n"
        + "    False: 1.0000\n"
    )
    assert out == expected
