import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim.models import Word2Vec
import numpy as np
import utils


class Create_Word2Vec:
    # Function to preprocess lyrics
    def preprocess_lyrics(self, lyrics):
        stop_words = set(stopwords.words('english'))
        processed_lyrics = []

        for lyric in lyrics:
            # Tokenize
            words = word_tokenize(lyric.lower())
            # Remove punctuation and stopwords
            words = [word for word in words if word.isalpha() and word not in stop_words]
            processed_lyrics.append(words)

        return processed_lyrics

    # Apply preprocessing to the lyrics column

    # Function to get the average vector for a song's lyrics
    def get_average_vector(self, words, model):
        word_vectors = [model.wv[word] for word in words if word in model.wv]
        if not word_vectors:  # Handle the case where no words are in the vocabulary
            return np.zeros(model.vector_size)
        return np.mean(word_vectors, axis=0)

    # Apply the function to each song's processed lyrics

    def run(self):
        df = pd.read_excel(utils.get_output_directory_path() + '/lyrics_output_genius.xlsx')
        df['processed_lyrics'] = self.preprocess_lyrics(df['lyrics'])

        # Train Word2Vec_Vectorizer model on all processed lyrics
        all_lyrics = df['processed_lyrics'].tolist()
        model = Word2Vec(sentences=all_lyrics, vector_size=20, window=5, min_count=1, workers=4)

        # Save the model (optional)
        model.save("lyrics_word2vec.model")

        df['lyrics_vector'] = df['processed_lyrics'].apply(lambda words: self.get_average_vector(words, model))

        # Create a new DataFrame with song names and their corresponding vectors
        final_df = pd.DataFrame({
            'song_name': df['songs'],
            'lyrics_vector': df['lyrics_vector']
        })

        # Convert lyrics_vector column to separate columns for each dimension
        vector_df = pd.DataFrame(final_df['lyrics_vector'].tolist(), index=final_df.index)
        final_df = pd.concat([final_df['song_name'], vector_df], axis=1)

        # Save the final DataFrame to a CSV file
        final_df.to_excel(utils.get_output_directory_path() + '/word2vec_lyrics_df.xlsx', index=False)

        return final_df
