from math import log
import os
import string
import nltk
import sys

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    retval = {}
    path = os.path.join(directory)
    for filename in os.listdir(path):
        with open(os.path.join(path, filename)) as f:
            retval[filename] = f.read()

    return retval


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    return [
        word.lower()
        for word in nltk.tokenize.word_tokenize(document)
        if word not in string.punctuation
        and word not in nltk.corpus.stopwords.words("english")
    ]


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    retval = {}
    num_of_documents = len(documents)
    words = set(
        [
            word
            for word in
            [
                value
                for values in documents.values()
                for value in values
            ]
        ]
    )

    for word in words:
        count = 0
        for document_words in documents.values():
            if word in document_words:
                count = count + 1
        retval[word] = log(num_of_documents / count)

    return retval


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    tf_idfs = {
        filename: 0
        for filename in files
    }
    for filename in files:
        for word in query:
            tf_idfs.update(
                {
                    filename: tf_idfs[filename] +
                    (files[filename].count(word) * idfs[word])
                }
            )

    return [
        filename
        for filename in sorted(
            tf_idfs, key=tf_idfs.get, reverse=True
        )
    ][:n]


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    matching_word_measures = {
        sentence: (0, 0)   # tuple(matching word measure, query term density)
        for sentence in sentences
    }
    for sentence in sentences:
        for word in set(sentences[sentence]):
            if word in query:
                matching_word_measures.update({
                    sentence: (
                        matching_word_measures[sentence][0] + idfs[word],
                        matching_word_measures[sentence][1] + (
                            sentences[sentence].count(word)
                            / len(sentences[sentence])
                        )
                    )
                })

    return [
        sentence
        for sentence in sorted(
            matching_word_measures, key=matching_word_measures.get, reverse=True
        )
    ][:n]


if __name__ == "__main__":
    main()
