import nltk
import sys
import os
import string
import math

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
    data = dict()

    for file in os.listdir(directory):
        with open(os.path.join(directory,file), encoding="Latin-1") as f:
            data[file] = f.read().replace("\n", " ")
    return data


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    words = []

    # Remove punctuation
    text = document.translate(str.maketrans('', '', string.punctuation))

    # Remove stopwords
    for word in nltk.word_tokenize(text.lower()):
        if not word in nltk.corpus.stopwords.words("english"):
            words.append(word)
    return words


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    words = set()
    idfs = dict()

    # Get all words in documents
    for document in documents:
        words.update(documents[document])

    # Calculate IDFs
    for word in words:
        f = sum(word in documents[document] for document in documents)
        idf = math.log(len(documents) / f)
        idfs[word] = idf
    return idfs


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    file_tfidf = dict()

    # Sum of tf-idf values for any word in the query for each file
    for file in files:
        total_tfidf = 0
        for word in query:
            total_tfidf += files[file].count(word) * idfs[word]
        file_tfidf[file] = total_tfidf

    # Sort files by its total tf-idf
    sorted_files = sorted(file_tfidf.items(), key=lambda x: x[1], reverse=True)
    return [file[0] for file in sorted_files][:n]


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    s = dict()

    # Calculate the total IDF and the query term density for each sentence
    for sentence in sentences:
        total_idf = 0
        words_in_query = 0
        for word in query:
            if word in sentences[sentence]:
                total_idf += idfs[word]
                words_in_query += 1
        density = words_in_query/len(sentences[sentence])
        s[sentence] = (total_idf,density)

    # Rank sentences acording to matching word measure and query term density
    sorted_s = sorted(s, key=lambda k: (s[k][0], s[k][1]), reverse=True)
    return sorted_s[:n]


if __name__ == "__main__":
    main()
