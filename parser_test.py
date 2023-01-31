import pytest
from nltk import Tree
from parser import preprocess, np_chunk


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


@pytest.mark.parametrize(
    "tree,expected",
    [
        (
            Tree.fromstring("(S (NP (N holmes)) (VP (V sat)))"),
            [
                Tree.fromstring("(NP (N holmes))")
            ]
        ),
        (
            Tree.fromstring(
                "(S (NP (N holmes)) (VP (V lit) (NP (Det a) (N pipe))))"),
            [
                Tree.fromstring("(NP (N holmes))"),
                Tree.fromstring("(NP (Det a) (N pipe))"),
            ]
        ),
        (
            Tree.fromstring(
                "(S\
                    (NP (N we))\
                    (VP\
                        (V arrived)\
                        (NP (Det the) (NP (N day) (NP (P before) (NP (N thursday)))))\
                    )\
                )"
            ),
            [
                Tree.fromstring("(NP (N we))"),
                Tree.fromstring("(NP (N thursday))")
            ]
        )
    ]

)
def test_np_chunk(tree: Tree, expected: list[Tree]):
    """
    It should accept a tree representing the syntax of a sentence, and return a list of all of the noun phrase chunks in that sentence.
    * For this problem, a “noun phrase chunk” is defined as a noun phrase that doesn’t contain other noun phrases within it. Put more formally, a noun phrase chunk is a subtree of the original tree whose label is NP and that does not itself contain other noun phrases as subtrees.
        * For example, if "the home" is a noun phrase chunk, then "the armchair in the home" is not a noun phrase chunk, because the latter contains the former as a subtree.
    * You may assume that the input will be a nltk.tree object whose label is S (that is to say, the input will be a tree representing a sentence).
    * Your function should return a list of nltk.tree objects, where each element has the label NP.
    """
    output = np_chunk(tree)
    assert output == expected
