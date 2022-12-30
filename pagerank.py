import os
import random
import re
import sys
import copy

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    retval = {}
    links_len = len(corpus[page])
    corpus_len = len(corpus)
    
    if links_len == 0:
        for page_name in corpus:
            retval[page_name] = 1 / corpus_len
        return retval

    random_probability = (1 - damping_factor) / corpus_len
    for page_name in corpus:
        retval[page_name] = random_probability
    
    for page_link in corpus[page]:
        pr = retval[page_link] + (damping_factor / links_len)
        retval.update({page_link: pr})
    
    return retval


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    track_visited = {}
    transition_weights = {}
    counter = 0
    pages = list(corpus.keys())

    for page_name in pages:
        track_visited[page_name] = 0
        
        model = transition_model(corpus, page_name, damping_factor)
        weights = []
        for page_name_model in pages:
            weights.append(model[page_name_model])
        
        transition_weights[page_name] = weights


    sample_page = random.choice(pages)
    
    while counter < n:
        counter = counter + 1
        track_visited.update({sample_page: track_visited[sample_page] + 1})
        sample_page = random.choices(pages, weights=transition_weights[sample_page])[0]
    
    retval = {}
    for page_name in pages:
        retval[page_name] = track_visited[page_name] / n

    return retval


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    retval = {}
    previous_PR = {}    
    linked_pages = {}
    pages = list(corpus.keys())

    for page_name in pages:
        retval[page_name] = 1 / len(corpus)
        previous_PR[page_name] = 0
        linked_pages[page_name] = set()

    for page in corpus:
        for link in corpus[page]:
            linked_pages[link].add(page)

    while not is_converged(previous_PR, retval):
        previous_PR = copy.deepcopy(retval)
        for page_name in pages:
            incoming_pages_probability = 0
            for link in linked_pages[page_name]:
                incoming_pages_probability = incoming_pages_probability + (previous_PR[link] / len(corpus[link]))

            retval[page_name] = ((1 - damping_factor) / len(corpus)) + (damping_factor * incoming_pages_probability)

    return retval

def is_converged(previous: dict, current: dict, accuracy=0.001) -> bool:
    if previous.keys() != current.keys():
        raise ValueError("Cannot calculate accuracy for dictionaries with incompatible keys")
    
    for page_name in previous:
        if abs(previous[page_name] - current[page_name]) > accuracy:
            return False
    return True


if __name__ == "__main__":
    main()
