import pytest
from questions import *


def test_load_files():
    output = load_files("test_corpus")
    expected = {
        "test.txt": "Test\nfile"
    }

    assert output == expected
