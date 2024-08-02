import pandas as pd
from openpyxl.reader.excel import load_workbook
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, \
    RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LinearRegression, SGDRegressor, LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.metrics import accuracy_score, root_mean_squared_error, mean_absolute_error, mean_squared_error, r2_score, \
    confusion_matrix, precision_score, recall_score, f1_score, matthews_corrcoef
from sklearn.model_selection import KFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

import utils
import sklearn


class AudioProcessor:

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

    def evaluation_regressor(self, target_df, features_df, column_to_predict, model, test_size=0.2):

        song_features_df = pd.read_excel(utils.get_output_directory_path() + '/song_features_output.xlsx')

        # Split dataset into features (X) and target (y)
        X = features_df.drop(columns="songs")  # Features
        y = target_df[column_to_predict]  # Target

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # Initialize your model (example: Linear Regression)
        # model = LinearRegression()

        # Initialize KFold cross-validation (10 folds)
        kf = KFold(n_splits=10, shuffle=True, random_state=42)

        # Perform cross-validation on the training set
        train_rmses = []
        train_maes = []
        train_mses = []
        train_r2s = []
        test_rmses = []
        test_maes = []
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
            train_rmse = root_mean_squared_error(y_train_fold, y_train_pred)
            train_mae = mean_absolute_error(y_train_fold, y_train_pred)
            train_mse = mean_squared_error(y_train_fold, y_train_pred)
            train_r2 = r2_score(y_train_fold, y_train_pred)
            train_rmses.append(train_rmse)
            train_maes.append(train_mae)
            train_mses.append(train_mse)
            train_r2s.append(train_r2)

            # Predict on the validation fold (test set for cross-validation)
            y_val_pred = model.predict(X_val_fold)

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
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
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
            'Test RMSE': [final_test_rmse],
            'Test MAE': [final_test_mae],
            'Test MSE': [final_test_mse],
            'Test R^2': [final_test_r2]
        })

        # Print to excel
        model_name = type(model).__name__

        # Define the path to the existing Excel file
        excel_filename = utils.get_output_directory_path() + '/audio_evaluation_regressor.xlsx'

        self.append_df_to_excel(excel_filename, results_df, model_name)

    def evaluation_classifier(self, extracted_audio_features_df, target_df, column_to_predict, model, test_size=0.2):

        # Selecting relevant columns
        # Split dataset into features (X) and target (y)
        X = extracted_audio_features_df  # Features
        y = target_df[column_to_predict]  # Target

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
        # roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]) if hasattr(model, "predict_proba") else "N/A"
        # logloss = log_loss(y_test, model.predict_proba(X_test)[:, 1]) if hasattr(model, "predict_proba") else "N/A"
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
            # 'ROC AUC': [roc_auc],
            # 'Log Loss': [logloss],
            'MCC': [mcc]
        })

        # Print results to Excel
        excel_filename = utils.get_output_directory_path() + '/audio_evaluation_classifier.xlsx'
        model_name = type(model).__name__

        self.append_df_to_excel(excel_filename, results_df, model_name)

    def run(self):
        print("Processing lyrics!\n")

        print("Loading dataset!\n")
        # Read input DataFrame
        song_features_df = pd.read_excel(utils.get_output_directory_path() + '/song_features_output.xlsx')

        original_df = pd.read_excel(utils.get_output_directory_path() + '/skladba_dataframe_cleaned.xlsx')

        # Get columns for audio
        # audio_columns = [
        #    col for col in original_df.columns
        #    if any(keyword in col.lower() for keyword in ['songs', 'rhythm', 'melody', 'harmony'])
        # ]

        columns_to_predict = [
            col for col in original_df.columns
            if col.lower().endswith('_mean') and (
                    col.lower().startswith('rhythm') or
                    col.lower().startswith('melody') or
                    col.lower().startswith('harmony'))

        ]

        audio_columns_df = original_df[columns_to_predict]
        #merged = utils.merge_dataframes(song_features_df, audio_columns_df,
        #                               "merged_audio_for_prefiction_df.xlsx", "songs")

        # columns_to_predict = [col for col in audio_columns_df.columns if col.lower().endswith('_mean') and (col.lower() in ['songs', 'rhythm', 'melody', 'harmony'])]
        # columns_to_predict = ['rhythm_mean', 'harmony_mean', 'melody_mean']

        # Models that are being used in ML algorithm
        models = [
            RandomForestRegressor(random_state=42),
            GradientBoostingRegressor(random_state=42),
            ExtraTreesRegressor(random_state=42),
            DecisionTreeRegressor(random_state=42),
            LinearRegression(),
            SVR(),
            SGDRegressor(random_state=42),
            GaussianProcessRegressor(random_state=42)
        ]

        classifiers = [
            SVC(kernel='linear', random_state=42),
            RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42),
            LogisticRegression(C=1.0, penalty='l2', random_state=42),
            RidgeClassifier(alpha=1.0, random_state=42),
            SGDClassifier(alpha=0.0001, penalty='l2', loss='log_loss', random_state=42),
            DecisionTreeClassifier(max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=42),
            KNeighborsClassifier(n_neighbors=10)  # Not sure for parameters
            # NaiveBayesClassifier()
        ]
        param_grids_regressor = [
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
                'C': [0.1, 1, 10],  # C can be between 0.1 and 10
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

        # print("Hyperparameters tuning!")
        # for model, param_grid in zip(models, param_grids_regressor):
        #    for column in columns_to_predict:
        #        self.find_optimal_hyperparameters(merged, column, model, param_grid)

        song_features_df = pd.read_excel(utils.get_output_directory_path() + '/song_features_output.xlsx')

        target_df = audio_columns_df#.drop(columns=['songs'])

        print("Evaluating Audio with Regressors!\n")
        for model, param_grid in zip(models, param_grids_regressor):
            for column in columns_to_predict:
                self.evaluation_regressor(target_df, song_features_df, column, model)

        # print("Evaluating Audio with Classifiers!\n")
        # classification_df = utils.create_classifications(audio_columns_df.copy())
        # target_columns = [col for col in classification_df.columns if col.lower().endswith('_mean')]
        #
        # merged = utils.merge_dataframes(classification_df, song_features_df)
        #
        # columns_to_predict = [col for col in classification_df.columns if col.lower().endswith('class')]
        # target_df = merged[columns_to_predict]
        #
        # features_df = merged.drop(target_columns + columns_to_predict + ['songs'], axis=1)
        #
        # for model in classifiers:
        #     for column in columns_to_predict:
        #         self.evaluation_classifier(features_df, target_df, column, model)
