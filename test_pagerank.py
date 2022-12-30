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
            pytest.approx({
                "1.html": 0.05,
                "2.html": 0.475,
                "3.html": 0.475
            }, 0.001)
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



@pytest.mark.parametrize(
    "corpus,damping_factor,n,expected",
    [
        (
            {
                "1.html": {"2.html", "3.html"}, 
                "2.html": {"3.html"}, 
                "3.html": {"2.html"}
            },
            0.85,
            10000,
            pytest.approx({
                "1.html": 0.05,
                "2.html": 0.475,
                "3.html": 0.475
            }, abs=0.005)
        ),
        (
            {
                "1.html": {}, 
                "2.html": {}, 
                "3.html": {}
            },
            0.85,
            10000,
            pytest.approx({
                "1.html": 0.333,
                "2.html": 0.333,
                "3.html": 0.333
            }, abs=0.01)
        )
    ]
)
def test_sample_pagerank(corpus, damping_factor, n, expected):
    """
    It should return estimated PageRank for each page.
    """    
    output = sample_pagerank(corpus, damping_factor, n)
    
    assert output == expected
    
    sum = 0
    for page_rank in list(output.keys()):
        sum = sum + output[page_rank]
    assert sum == 1 # values in output should sum up to 1


@pytest.mark.parametrize(
    "corpus,damping_factor,expected",
    [
        (
            {
                "1.html": {"2.html", "3.html"}, 
                "2.html": {"3.html"}, 
                "3.html": {"2.html"}
            },
            0.85,
            pytest.approx({
                "1.html": 0.05,
                "2.html": 0.475,
                "3.html": 0.475
            }, 0.001)
        ),
        (
            {
                "1.html": {}, 
                "2.html": {"3.html"}, 
                "3.html": {"2.html"}
            },
            0.85,
            pytest.approx({
                "1.html": 0.0699,
                "2.html": 0.465,
                "3.html": 0.465
            }, 0.001)
        )
    ]
)
def test_iterate_pagerank(corpus, damping_factor, expected):
    """
    It should return each page's PageRank accurate to within 0.001.
    """    
    output = iterate_pagerank(corpus, damping_factor)
    
    assert output == expected
    
    sum = 0
    for page_rank in list(output.keys()):
        sum = sum + output[page_rank]
    assert sum == pytest.approx(1) # values in output should sum up to 1

    
