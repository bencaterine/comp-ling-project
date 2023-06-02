import numpy as np
import pandas as pd

movies_df = pd.read_csv('movies_df.csv')
similarities = np.loadtxt("similarities.csv", delimiter=",")

def get_most_similar(movie, n=10):
    # return the n most similar movies to the given movie
    if movie not in movies_df.Title.values:
        return 'Movie not found'
    
    movie_index = movies_df.index[movies_df.Title == movie][0]
    movie_sims = similarities[movie_index]
    ind = movie_sims.argsort()[-(n+1):-1][::-1]
    return pd.DataFrame({
        'movies': movies_df.Title.values[ind],
        'similarity': movie_sims[ind]
    })

if __name__ == '__main__':
    print(get_most_similar('Ant-Man'))