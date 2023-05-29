import numpy as np

class Embeddings:

    def __init__(self, glove_file='glove_top50k_50d.txt'):
        self.embeddings = {}
        self.word_rank = {}
        for idx, line in enumerate(open(glove_file)):
            row = line.split()
            word = row[0]
            vals = np.array([float(x) for x in row[1:]])
            self.embeddings[word] = vals
            self.word_rank[word] = idx + 1

    def __getitem__(self, word):
        return self.embeddings[word]

    def __contains__(self, word):
        return word in self.embeddings

    def vector_norm(self, vec):
        """
        Calculate the vector norm (aka length) of a vector.

        This is given in SLP Ch. 6, equation 6.8. For more information:
        https://en.wikipedia.org/wiki/Norm_(mathematics)#Euclidean_norm

        Parameters
        ----------
        vec : np.array
            An embedding vector.

        Returns
        -------
        float
            The length (L2 norm, Euclidean norm) of the input vector.
        """
        return np.sqrt(np.square(vec).sum())

    def cosine_similarity(self, v1, v2):
        """
        Calculate cosine similarity between v1 and v2; these could be
        either words or numpy vectors.

        If either or both are words (e.g., type(v#) == str), replace them 
        with their corresponding numpy vectors before calculating similarity.

        Parameters
        ----------
        v1, v2 : str or np.array
            The words or vectors for which to calculate similarity.

        Returns
        -------
        float
            The cosine similarity between v1 and v2.
        """
        if type(v1) == str:
            v1 = self[v1]
        if type(v2) == str:
            v2 = self[v2]
        return np.dot(v1, v2) / (self.vector_norm(v1) * self.vector_norm(v2))
