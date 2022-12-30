import pytest
from pagerank import *



@pytest.mark.parametrize(
    "corpus,page,damping_factor,expected",
    [
        (
            {
                "1.html": {"2.html", "3.html"}, 
                "2.html": {"3.html"}, 
                "3.html": {"2.html"}
            },
            "1.html",
            0.85,
            {
                "1.html": pytest.approx(0.05, 0.0001),
                "2.html": pytest.approx(0.475, 0.0001),
                "3.html": pytest.approx(0.475, 0.0001)
            }
        ),
        (
            {
                "1.html": {}, 
                "2.html": {"3.html"}, 
                "3.html": {"2.html"},
                "4.html": {"2.html"}
            },
            "1.html",
            0.85,
            {
                "1.html": 0.25,
                "2.html": 0.25,
                "3.html": 0.25,
                "4.html": 0.25
            }
        )
    ]
)
def test_transition_model(corpus, page, damping_factor, expected):
    """ 
    It should return probability distribution properly.
    """
    output = transition_model(corpus, page, damping_factor)

    assert output == expected

