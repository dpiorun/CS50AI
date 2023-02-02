import pytest
from questions import *


@pytest.fixture
def documents():
    return {
        "a.txt": ["test", "cat", "dog"],
        "b.txt": ["cat", "dog", "dog"],
        "c.txt": ["cat", "mouse"]
    }


@pytest.fixture
def idfs():
    return {
        "test": log(3 / 1),
        "cat": log(3 / 3),
        "dog": log(3 / 2),
        "mouse": log(3 / 1)
    }


def test_load_files():
    """
    It should accept the name of a directory and return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file’s contents as a string.
    * Your function should be platform-independent: that is to say, it should work regardless
    of operating system. Note that on macOS, the `/` character is used to separate path components,
    while the `\\` character is used on Windows. Use os.sep and os.path.join as needed instead
    of using your platform’s specific separator character.
    * In the returned dictionary, there should be one key named for each `.txt` file in the
    directory. The value associated with that key should be a string (the result of reading
    the corresonding file).
    * Each key should be just the filename, without including the directory name. For example,
    if the directory is called corpus and contains files `a.txt` and `b.txt`, the keys should be
    `a.txt` and `b.txt` and not `corpus/a.txt` and `corpus/b.txt`.
    """
    output = load_files("test_corpus")
    expected = {
        "test.txt": "Test\nfile"
    }

    assert output == expected


def test_tokenize():
    """
    It should accept a document (a string) as input, and return a list of all of the words in that
    document, in order and lowercased.
    * You should use `nltk’s` `word_tokenize` function to perform tokenization.
    * All words in the returned list should be lowercased.
    * Filter out punctuation and stopwords (common words that are unlikely to be useful for
    querying). Punctuation is defined as any character in string.punctuation (after you
    `import string`). Stopwords are defined as any word in `nltk.corpus.stopwords.words("english")`.
    * If a word appears multiple times in the document, it should also appear multiple times in the
    returned list (unless it was filtered out).
    """
    output = tokenize("Dog chases a cat, but not a mouse. Bad dog.")
    expected = [
        "dog", "chases", "cat", "mouse", "bad", "dog"
    ]
    assert output == expected


def test_compute_idfs(documents: dict[str, list[str]], idfs: dict[str, float]):
    """
    It should accept a dictionary of documents and return a new dictionary mapping words to their
    IDF (inverse document frequency) values.
    * Assume that documents will be a dictionary mapping names of documents to a list of words in
    that document.
    * The returned dictionary should map every word that appears in at least one of the documents
    to its inverse document frequency value.
    * Recall that the inverse document frequency of a word is defined by taking the natural
    logarithm of the number of documents divided by the number of documents in which the word appears.
    """
    output = compute_idfs(documents)

    assert output == idfs


@pytest.mark.parametrize(
    "query,n,expected",
    [
        (
            {"dog", "cat"},
            2,
            ["b.txt", "a.txt"]
        ),
        (
            {"dog", "cat", "mouse"},
            2,
            ["c.txt", "b.txt"]
        )
    ]
)
def test_top_files(
    query: set[str],
    n: int,
    expected,
    documents: dict[str, list[str]],
    idfs: dict[str, float]
):
    output = top_files(query, documents, idfs, n)
    assert expected == output
