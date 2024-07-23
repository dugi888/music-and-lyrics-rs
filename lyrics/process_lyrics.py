import pandas as pd
from openpyxl.reader.excel import load_workbook
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier, RidgeClassifier, SGDRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, \
    RandomForestClassifier
from sklearn.metrics import root_mean_squared_error, r2_score, accuracy_score, mean_squared_error, mean_absolute_error, \
    confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, log_loss, matthews_corrcoef
from nltk.stem import WordNetLemmatizer
from nltk import PorterStemmer, NaiveBayesClassifier
from sklearn import decomposition
import utils


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

    @staticmethod
    def extract_song_name(song):
        if ':' in song:
            return song.split(':')[1].strip()
        else:
            return song  # return as is if no ':' found

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
        df_combined.to_excel(utils.get_output_directory_path() + '/tf_idf_dataframe.xlsx', index=False)
        return df_combined

    def dimension_reduction(self, df):
        pca_data = self.pca.fit_transform(df)
        return pca_data, self.pca

    @staticmethod
    def append_df_to_excel(filename, df, sheet_name):
        try:
            # Load the existing workbook
            book = load_workbook(filename)
            if sheet_name in book.sheetnames:
                # Load existing sheet into a DataFrame
                existing_df = pd.read_excel(filename, sheet_name=sheet_name)
                # Append the new data without column names
                combined_df = pd.concat([existing_df, df], ignore_index=True)
            else:
                # If the sheet does not exist, just use the new DataFrame
                combined_df = df

            # Write to Excel
            with pd.ExcelWriter(filename, engine='openpyxl', mode='a') as writer:
                if sheet_name in writer.book.sheetnames:
                    writer.book.remove(writer.book[sheet_name])
                combined_df.to_excel(writer, sheet_name=sheet_name, index=False)
        except FileNotFoundError:
            # If the file doesn't exist, create a new one and write the DataFrame
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name=sheet_name, index=False)

    # TODO Optimize this method
    def evaluation_regressor(self, df, column_to_predict, model, param_grid, test_size=0.2):

        # Selecting relevant columns
        columns_to_select = [f'PC{i}' for i in range(1, 21)]
        selected_df = df[columns_to_select]

        # Split dataset into features (X) and target (y)
        X = selected_df  # Features
        y = df[column_to_predict]  # Target

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # Initialize your model (example: Linear Regression)
        # model = LinearRegression()

        # Initialize KFold cross-validation (10 folds)
        kf = KFold(n_splits=10, shuffle=True, random_state=42)

        # Find the best parameters for model
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=10)
        grid_search.fit(X, y)
        best_model = grid_search.best_estimator_
        # Perform cross-validation on the training set
        train_rmses = []
        train_maes = []
        train_mses = []
        train_r2s = []
        test_rmses = []
        test_maes = []
        test_mses = []
        test_r2s = []

        baseline_predictions = [y.mean()] * len(y)
        baseline_rmse = root_mean_squared_error(y, baseline_predictions)

        for train_index, test_index in kf.split(X_train):
            X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[test_index]
            y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[test_index]

            # Fit the model on the training fold
            best_model.fit(X_train_fold, y_train_fold)

            # Predict on the training fold
            y_train_pred = best_model.predict(X_train_fold)

            # Evaluate training fold
            train_rmse = root_mean_squared_error(y_train_fold, y_train_pred)
            train_mae = mean_absolute_error(y_train_fold, y_train_pred)
            train_mse = mean_squared_error(y_train_fold, y_train_pred)
            train_r2 = r2_score(y_train_fold, y_train_pred)
            train_rmses.append(train_rmse)
            train_maes.append(train_mae)
            train_mses.append(train_mse)
            train_r2s.append(train_r2)

            # Predict on the validation fold (test set for cross-validation)
            y_val_pred = best_model.predict(X_val_fold)

            # Evaluate validation fold
            test_rmse = root_mean_squared_error(y_val_fold, y_val_pred)
            test_mae = mean_absolute_error(y_val_fold, y_val_pred)
            test_mse = mean_squared_error(y_val_fold, y_val_pred)
            test_r2 = r2_score(y_val_fold, y_val_pred)
            test_rmses.append(test_rmse)
            test_maes.append(test_mae)
            test_mses.append(test_mse)
            test_r2s.append(test_r2)

        # Evaluate on the test set (final evaluation)
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)
        final_test_rmse = root_mean_squared_error(y_test, y_pred)
        final_test_mae = mean_absolute_error(y_test, y_pred)
        final_test_mse = mean_squared_error(y_test, y_pred)
        final_test_r2 = r2_score(y_test, y_pred)

        # Generating dataframe from results
        results_df = pd.DataFrame({
            'Feature': [column_to_predict],
            'Training RMSE': [sum(train_rmses) / len(train_rmses)],
            'Training MAE': [sum(train_maes) / len(train_maes)],
            'Training MSE': [sum(train_mses) / len(train_mses)],
            'Training R^2': [sum(train_r2s) / len(train_r2s)],
            'Baseline RMSE': [baseline_rmse],
            'Test RMSE': [final_test_rmse],
            'Test MAE': [final_test_mae],
            'Test MSE': [final_test_mse],
            'Test R^2': [final_test_r2],
            'RMSE dif': [baseline_rmse - final_test_rmse]
        })

        # Print to excel
        model_name = type(model).__name__

        # Define the path to the existing Excel file
        excel_filename = utils.get_output_directory_path() + '/model_evaluation_best_fit.xlsx'

        self.append_df_to_excel(excel_filename, results_df, model_name)


    @staticmethod
    def find_optimal_hyperparameters(df, column_to_predict, model, param_grid):
        columns_to_select = [f'PC{i}' for i in range(1, 21)]
        selected_df = df[columns_to_select]

        # Split dataset into features (X) and target (y)
        X = selected_df  # Features
        y = selected_df[column_to_predict]  # Target

        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=10, scoring='neg_root_mean_squared_error')
        grid_search.fit(X, y)

        # Print the best parameters and best score
        best_params = (f"Best parameters for {type(model).__name__} when predicting {column_to_predict}:\n" +
                       str(grid_search.best_params_) + '\n')
        best_result = f"Best neg_root_mean_squared_error score: {grid_search.best_score_}\n"

        # Check if file exists
        filename = utils.get_output_directory_path() + '/hyper_parameters_for_regressors.txt'

        with open(filename, 'a') as file:
            file.write(best_params + best_result + "\n\n")

        print(f"Best parameters for {type(model).__name__} when predicting {column_to_predict}:")
        print(grid_search.best_params_)
        print(f"Best neg_root_mean_squared_error score: {grid_search.best_score_}\n")

    def evaluation_classifier(self, df, column_to_predict, model, test_size=0.2):

        # Selecting relevant columns
        columns_to_select = [column_to_predict] + [f'PC{i}' for i in range(1, 21)]
        selected_df = df[columns_to_select]

        # Split dataset into features (X) and target (y)
        X = selected_df.drop(columns=[column_to_predict])  # Features
        y = selected_df[column_to_predict]  # Target

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # Initialize KFold cross-validation (10 folds)
        kf = KFold(n_splits=10, shuffle=True, random_state=42)

        # Perform cross-validation on the training set
        train_scores = []
        test_scores = []

        for train_index, test_index in kf.split(X_train):
            X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[test_index]
            y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[test_index]

            # Fit the model on the training fold
            model.fit(X_train_fold, y_train_fold)

            # Predict and evaluate on the training fold
            y_train_pred = model.predict(X_train_fold)
            train_score = accuracy_score(y_train_fold, y_train_pred)
            train_scores.append(train_score)

            # Predict and evaluate on the validation fold
            y_val_pred = model.predict(X_val_fold)
            test_score = accuracy_score(y_val_fold, y_val_pred)
            test_scores.append(test_score)

        # Evaluate on the test set (final evaluation)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Evaluation
        final_test_score = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        #roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]) if hasattr(model, "predict_proba") else "N/A"
        #logloss = log_loss(y_test, model.predict_proba(X_test)[:, 1]) if hasattr(model, "predict_proba") else "N/A"
        mcc = matthews_corrcoef(y_test, y_pred)

        # Calculate average training and validation scores
        avg_train_score = sum(train_scores) / len(train_scores)
        avg_test_score = sum(test_scores) / len(test_scores)

        # Prepare results DataFrame
        results_df = pd.DataFrame({
            'Feature': [column_to_predict],
            'Training Accuracy': [avg_train_score],
            'Validation Accuracy': [avg_test_score],
            'Test Accuracy': [final_test_score],
            'Precision': [precision],
            'Recall': [recall],
            'F1-Score': [f1],
            #'ROC AUC': [roc_auc],
            #'Log Loss': [logloss],
            'MCC': [mcc]
        })

        # Print results to Excel
        excel_filename = utils.get_output_directory_path() + '/model_evaluation_classifier.xlsx'
        model_name = type(model).__name__

        self.append_df_to_excel(excel_filename, results_df, model_name)

    def create_classifications(self, df):
        for col in df.columns:
            if col.lower() == 'songs':
                continue
            one_third_value, two_thirds_value = utils.find_boundaries(df[col])
            df.loc[:, col + '_class'] = df[col].apply(
                lambda x: utils.classify_value(x, one_third_value, two_thirds_value))
        df.to_excel(utils.get_output_directory_path() + '/lyrics_features_classified.xlsx', index=False)
        return df

    def run(self):
        print("Processing lyrics!\n")

        print("Loading dataset!\n")
        # Read input DataFrame
        df = pd.read_excel(utils.get_output_directory_path() + '/lyrics_output_genius.xlsx')

        # Calculate TF-IDF
        print("Calculating TF-IDF!\n")
        tfidf_df = pd.read_excel(utils.get_output_directory_path() + '/tf_idf_dataframe.xlsx') # self.calculate_tfidf(df)

        # Perform dimension reduction
        print("Reducing dimensions!\n")
        send_df = tfidf_df.iloc[:, 4:]  # Selecting all columns from index 4 to the end
        pca_data, pca = self.dimension_reduction(send_df)

        # Create a DataFrame with the PCA data
        pca_df = pd.DataFrame(pca_data, columns=[f'PC{i + 1}' for i in range(pca_data.shape[1])])

        # Include the original song titles in the PCA DataFrame
        pca_df = pd.concat([tfidf_df[['songs']], pca_df], axis=1)
        pca_df.to_excel(utils.get_output_directory_path() + '/PCA_dataframe.xlsx', index=False)

        # Prediction
        pca_df = pd.read_excel(utils.get_output_directory_path() + '/PCA_dataframe.xlsx')

        original_df = pd.read_excel(utils.get_output_directory_path() + '/skladba_dataframe_cleaned.xlsx') # , converters={'songs': self.extract_song_name}
        lyrics_columns = [col for col in original_df.columns if
                          col.lower().startswith('lyrics_') and col.lower().endswith('_mean') or 'songs' in col.lower()]

        lyrics_features_df = original_df[lyrics_columns]

        # TODO Check output_file_name, try to change it to be more general
        merged = utils.merge_dataframes(pca_df, lyrics_features_df, "merged_dataframe.xlsx")

        columns_to_predict = [col for col in lyrics_features_df.columns if 'lyrics' in col.lower()]
        # ['lyrics_complexity_mean', 'lyrics_comprehensibility_mean', 'lyrics_depth_mean']

        # Models that are being used in ML algorithm
        models = [
            RandomForestRegressor(random_state=42),
            GradientBoostingRegressor(random_state=42),
            ExtraTreesRegressor(random_state=42),
            DecisionTreeRegressor(random_state=42),
            LinearRegression(),
            SVR(),
            SGDRegressor(random_state=42),
            GaussianProcessRegressor()
        ]

        classifiers = [
            SVC(kernel='linear'),
            RandomForestClassifier(n_estimators=100, max_depth=None),
            LogisticRegression(C=1.0, penalty='l2', random_state=42),
            RidgeClassifier(alpha=1.0, random_state=42),
            SGDClassifier(alpha=0.0001, penalty='l2', loss='log_loss', random_state=42),
            DecisionTreeClassifier(max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=42),
            KNeighborsClassifier(n_neighbors=10)  # Not sure for parameters
            # NaiveBayesClassifier() How to implement this and fix parameters.
        ]
        param_grids = [
            {  # RandomForestRegressor parameters
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            },
            {  # GradientBoostingRegressor parameters
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.05, 0.1, 0.2],
                'max_depth': [3, 5],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            },
            {  # ExtraTreesRegressor parameters
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            },
            {  # DecisionTreeRegressor parameters
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            },
            {  # LinearRegression has no hyperparameters to tune
                'fit_intercept': [True, False]
            },
            {  # SVC parameters
                'C': [0.1, 0.8, 2, 5, 7, 10],  # C can be between 0.1 and 10
                'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                'degree': [2, 3, 4, 5],  # Degrees 2 to 5 for the polynomial kernel
                'gamma': ['scale', 'auto'],  # Common choices for gamma
                'epsilon': [0.01, 0.05, 0.1, 0.4, 0.6, 1],  # Epsilon can be between 0.01 and 1
                'shrinking': [True, False]  # Whether to use the shrinking heuristic
            },
            {  # SGD parameters
                'loss': ['squared_error', 'huber', 'epsilon_insensitive'],
                'penalty': ['none', 'l2', 'l1', 'elasticnet'],
                'alpha': [1, 0.1, 1e-4, 1e-2],
                'fit_intercept': [True, False],
                'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'],
                'eta0': [1e-3, 1e-2, 1e-1],
                'power_t': [0.25, 0.5, 0.75]
            }
        ]

        print("Evaluating Regressors!\n")
        # for model, param_grid in zip(models, param_grids):
        #     for column in columns_to_predict:
        #         self.evaluation_regressor(merged, column, model, param_grid)

        classification_df = self.create_classifications(lyrics_features_df.copy())
        columns_to_predict = [col for col in classification_df.columns if col.lower().endswith('class')]
        merged = utils.merge_dataframes(classification_df, pca_df)
        print("Evaluating Classifiers!\n")
        for model in classifiers:
            for column in columns_to_predict:
                self.evaluation_classifier(merged, column, model)
