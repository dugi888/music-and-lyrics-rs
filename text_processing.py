import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier, RidgeClassifier, SGDRegressor
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, r2_score
from nltk.stem import WordNetLemmatizer
from nltk import PorterStemmer
from sklearn import decomposition


# HAD TO DOWNLOAD THIS FOR THE FIRST TIME
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
class TextProcessing:
    def __init__(self, use_stemming=True, n_components=20):
        print('-- Creating TextProcessing --')
        self.use_stemming = use_stemming
        self.n_components = n_components
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.pca = decomposition.PCA(n_components=self.n_components)

    def preprocess_lyrics(self, lyrics):
        tokens = word_tokenize(lyrics)
        if self.use_stemming:
            processed_tokens = [self.stemmer.stem(token) for token in tokens]
        else:
            processed_tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        return ' '.join(processed_tokens)

    def calculate_tfidf(self, df):
        # Preprocess text
        df['processed_lyrics'] = df['lyrics'].apply(self.preprocess_lyrics)

        # Initialize the TF-IDF Vectorizer
        tfidf_vectorizer = TfidfVectorizer()

        # Fit and transform the lyrics
        tfidf_matrix = tfidf_vectorizer.fit_transform(df['processed_lyrics'])

        # Create a DataFrame from the TF-IDF matrix
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

        # Combine the original DataFrame with the TF-IDF DataFrame
        df_combined = pd.concat([df, tfidf_df], axis=1)

        # Save the combined DataFrame to an Excel file
        df_combined.to_excel('output_tables/tf_idf_dataframe.xlsx', index=False)
        return df_combined

    def dimension_reduction(self, df):
        pca_data = self.pca.fit_transform(df)
        return pca_data, self.pca

    @staticmethod
    def merge_dataframes_by_song_name(pca_df, original_df):
        merged_df = pd.merge(original_df, pca_df, on="songs", how="inner")
        return merged_df

    def regression_algorithm(self, df, column_to_predict, model, test_size=0.2):

        # Selecting relevant columns
        columns_to_select = [column_to_predict] + [f'PC{i}' for i in range(1, 21)]
        selected_df = df[columns_to_select]

        # Split dataset into features (X) and target (y)
        X = selected_df.drop(columns=[column_to_predict])  # Features
        y = selected_df[column_to_predict]  # Target

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # Initialize your model (example: Linear Regression)
        # model = LinearRegression()

        # Initialize KFold cross-validation (10 folds)
        kf = KFold(n_splits=10, shuffle=True, random_state=42)

        # Perform cross-validation on the training set
        train_mses = []
        train_r2s = []
        test_mses = []
        test_r2s = []

        for train_index, test_index in kf.split(X_train):
            X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[test_index]
            y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[test_index]

            # Fit the model on the training fold
            model.fit(X_train_fold, y_train_fold)

            # Predict on the training fold
            y_train_pred = model.predict(X_train_fold)

            # Evaluate training fold
            train_mse = mean_squared_error(y_train_fold, y_train_pred)
            train_r2 = r2_score(y_train_fold, y_train_pred)
            train_mses.append(train_mse)
            train_r2s.append(train_r2)

            # Predict on the validation fold (test set for cross-validation)
            y_val_pred = model.predict(X_val_fold)

            # Evaluate validation fold
            test_mse = mean_squared_error(y_val_fold, y_val_pred)
            test_r2 = r2_score(y_val_fold, y_val_pred)
            test_mses.append(test_mse)
            test_r2s.append(test_r2)

        # Evaluate on the test set (final evaluation)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        final_test_mse = mean_squared_error(y_test, y_pred)
        final_test_r2 = r2_score(y_test, y_pred)

        # Print results

        model_name = type(model).__name__
        print(column_to_predict, " & ", model_name, '\n --------------------------- \n')
        print("Training Set:")
        print(f"Mean Squared Error (MSE): {sum(train_mses) / len(train_mses):.10f}")
        print(f"R^2 Score: {sum(train_r2s) / len(train_r2s):.10f}")
        print("\nCross-Validation (Validation Set):")
        print(f"Mean Cross-Validation MSE: {sum(test_mses) / len(test_mses):.10f}")
        print(f"Mean Cross-Validation R^2 Score: {sum(test_r2s) / len(test_r2s):.10f}")
        print("\nTest Set (Final Evaluation):")
        print(f"Final Test MSE: {final_test_mse:.10f}")
        print(f"Final Test R^2 Score: {final_test_r2:.10f}")
        print("\n\n")

    @staticmethod
    def merge_dataframes(pca_df, original_df):
        merged_df = pd.merge(original_df, pca_df, on="songs", how="inner")
        merged_df.to_excel('output_tables/merged_dataframe.xlsx', index=False)
        return merged_df

    @staticmethod
    def extract_song_name(song):
        if ':' in song:
            return song.split(':')[1].strip()
        else:
            return song  # return as is if no ':' found

    def run(self):
        # Read input DataFrame
        df = pd.read_excel("output_tables/lyrics_output.xlsx")

        # Calculate TF-IDF
        full_df = self.calculate_tfidf(df)

        # Perform dimension reduction
        send_df = full_df.iloc[:, 4:]  # Selecting all columns from index 4 to the end
        pca_data, pca = self.dimension_reduction(send_df)

        # Create a DataFrame with the PCA data
        pca_df = pd.DataFrame(pca_data, columns=[f'PC{i + 1}' for i in range(pca_data.shape[1])])

        # Include the original song titles in the PCA DataFrame
        pca_df = pd.concat([full_df[['songs']], pca_df], axis=1)
        pca_df.to_excel('output_tables/PCA_dataframe.xlsx', index=False)

        # Prediction

        pca_df = pd.read_excel("output_tables/PCA_dataframe.xlsx")

        original_df = pd.read_excel("output_tables/skladba_dataframe_cleaned.xlsx",
                                    converters={'songs': self.extract_song_name})
        lyrics_columns = [col for col in original_df.columns if 'lyrics' in col.lower() or 'songs' in col.lower()]

        original_df_parameter = original_df[lyrics_columns]
        merged = self.merge_dataframes(pca_df, original_df_parameter)

        columns_to_predict = ['lyrics_complexity_mean', 'lyrics_comprehensibility_mean', 'lyrics_complexity_mean']

        models = [
            RandomForestRegressor(n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1,
                                  random_state=42),
            GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, min_samples_split=2,
                                      min_samples_leaf=1, random_state=42),
            ExtraTreesRegressor(n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1,
                                random_state=42),
            DecisionTreeRegressor(max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=42),
            LinearRegression()
            # LogisticRegression(C=1.0, penalty='l2', random_state=42),
            # SGDRegressor(alpha=0.0001, penalty='l2', loss='squared_loss', random_state=42),
            # RidgeClassifier(alpha=1.0, random_state=42),
            # SGDClassifier(alpha=0.0001, penalty='l2', loss='log', random_state=42),
            # DecisionTreeClassifier(max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=42)
        ]

        for column in columns_to_predict:
            for model in models:
                self.regression_algorithm(merged, column, model)
