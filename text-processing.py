import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# HAD TO DOWNLOAD THIS FOR THE FIRST TIME
#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('wordnet')

with open("song-lyrics/Amy_Winehouse-Rehab.txt", "r") as file:
    dna = file.read().replace("\n", " ")

def textProcess():
    # Example text
    text = "This is an example sentence. This sentence is for TF-IDF example."
    with open("song-lyrics/Amy_Winehouse-Rehab.txt", "r") as file:
        text = file.read().replace("\n", " ")
    # Text preprocessing
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens]
    tokens = [word for word in tokens if word.isalnum()]

    # Removing stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Lemmatizer
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # Join tokens back into string
    preprocessed_text = " ".join(tokens)

    # List of documents (example with one document)
    documents = [preprocessed_text]

    # TF-IDF Vectorizer
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)

    # Convert to DataFrame
    feature_names = vectorizer.get_feature_names_out()
    dense = tfidf_matrix.todense()
    dense_list = dense.tolist()
    df = pd.DataFrame(dense_list, columns=feature_names)
    #df.to_excel('testDF.xlsx', index=False)
    print(df)

