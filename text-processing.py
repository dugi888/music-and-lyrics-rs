import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import PorterStemmer
from sklearn import decomposition


# HAD TO DOWNLOAD THIS FOR THE FIRST TIME
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

def preprocess_lyrics(lyrics, use_stemming=True):
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    tokens = word_tokenize(lyrics)
    if use_stemming:
        processed_tokens = [stemmer.stem(token) for token in tokens]
    else:
        processed_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(processed_tokens)


def calculate_tfidf(df):
    # Preprocess text
    df['Processed_Lyrics'] = df['Lyrics'].apply(
        lambda x: preprocess_lyrics(x, use_stemming=True))  # True - stemmer, False - lemmatizer

    # Initialize the TF-IDF Vectorizer
    tfidf_vectorizer = TfidfVectorizer()

    # Fit and transform the lyrics
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['Processed_Lyrics'])

    # Create a DataFrame from the TF-IDF matrix
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

    # Combine the original DataFrame with the TF-IDF DataFrame
    df_combined = pd.concat([df, tfidf_df], axis=1)

    # Replace 0 with NaN
    df_combined.to_excel('output-tables/tf-idf-dataframe.xlsx', index=False)
    return df_combined


def dimension_reduction(df):
    pca = decomposition.PCA(n_components=20)
    pca_data = pca.fit_transform(df)
    return pca_data, pca


if __name__ == '__main__':
    df = pd.read_excel("output-tables/lyrics_output.xlsx")
    full_df = calculate_tfidf(df)
    send_df = full_df.iloc[:, 4:]
    pca_data, pca = dimension_reduction(send_df)

    # Create a DataFrame with the PCA data
    pca_df = pd.DataFrame(pca_data, columns=[f'PC{i + 1}' for i in range(pca_data.shape[1])])

    # Include the original song titles in the PCA DataFrame
    pca_df = pd.concat([full_df[['Song']], pca_df], axis=1)
    pca_df.to_excel('output-tables/PCA-dataframe.xlsx', index=False)

