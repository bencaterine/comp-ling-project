import numpy as np
import nltk
nltk.download('punkt')
from nltk import pos_tag
from nltk.tokenize import word_tokenize


def calculate_sentence_embedding(embeddings, sent, weighted = False):
    """
    Calculate a sentence embedding vector.

    If weighted is False, this is the elementwise sum of the constituent word vectors.
    If weighted is True, multiply each vector by a scalar calculated
    by taking the log of its word_rank. The word_rank value is available
    via a dictionary on the Embeddings class, e.g.:
       embeddings.word_rank['the'] # returns 1

    In either case, tokenize the sentence with the `word_tokenize` function,
    lowercase the tokens, and ignore any words for which we don't have word vectors. 

    Parameters
    ----------
    sent : str
        A sentence for which to calculate an embedding.

    weighted : bool
        Whether or not to use word_rank weighting.

    Returns
    -------
    np.array of floats
        Embedding vector for the sentence.
    
    """
    all_tokens = [w.lower() for w in word_tokenize(sent)]
    tokens = [t for t in all_tokens if t in embeddings]
    vecs = np.array([embeddings[token] for token in tokens])
    if weighted:
        ranks = np.array([embeddings.word_rank[token] for token in tokens])
        return np.dot(np.log(ranks), vecs)
    else:
        return vecs.sum(axis=0)
