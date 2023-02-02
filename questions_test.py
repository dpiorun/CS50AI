import pytest
from questions import *


def test_load_files():
    output = load_files("test_corpus")
    expected = {
        "test.txt": "Test\nfile"
    }

    assert output == expected


def test_tokenize():
    output = tokenize("Dog chases a cat, but not a mouse.")
    expected = [
        "dog", "chases", "cat", "mouse"
    ]
    assert output == expected
