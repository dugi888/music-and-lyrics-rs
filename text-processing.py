import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import PorterStemmer


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
    df_combined.replace(0, pd.NA, inplace=True)
    # df_combined.to_excel('testDF.xlsx', index=False)
    # print(df_combined)
