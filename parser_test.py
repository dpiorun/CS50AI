import pytest
from parser import preprocess


@pytest.mark.parametrize(
    "input_sentence,expected",
    [
        (
            "Holmes sat.",
            ["holmes", "sat"]
        ),
        (
            "Holmes lit 28 pipes.",
            ["holmes", "lit", "pipes"]
        )

    ]
)
def test_preprocess(input_sentence: str, expected: list[str]):
    """
        The preprocess function should accept a sentence as input and return a lowercased list of its words.
        * You may assume that sentence will be a string.
        * You should use nltk’s word_tokenize function to perform tokenization.
        * Your function should return a list of words, where each word is a lowercased string.
        * Any word that doesn’t contain at least one alphabetic character (e.g. . or 28) should be excluded from the returned list.
    """
    output = preprocess(input_sentence)
    assert output == expected
