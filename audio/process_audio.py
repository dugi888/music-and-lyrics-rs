import pandas as pd
from openpyxl.reader.excel import load_workbook
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, root_mean_squared_error, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

import utils
import sklearn


class AudioProcessor:

    def append_df_to_excel(self, filename, df, sheet_name):
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

    def evaluation(self, target_df, features_df, column_to_predict, model, test_size=0.2):

        song_features_df = pd.read_excel(utils.get_output_directory_path()+'/song_features_output.xlsx')

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
        excel_filename = utils.get_output_directory_path() + '/song_features_model_evaluation.xlsx'

        self.append_df_to_excel(excel_filename, results_df, model_name)

        # Print results
        # print(column_to_predict, " & ", model_name, '\n --------------------------- \n')
        # print("Training Set:")
        # print(f"Root Mean Squared Error (RMSE): {sum(train_mses) / len(train_mses):.10f}")
        # print(f"R^2 Score: {sum(train_r2s) / len(train_r2s):.10f}")
        # print("\nCross-Validation (Validation Set):")
        # print(f"Root Mean Cross-Validation RMSE: {sum(test_mses) / len(test_mses):.10f}")
        # print(f"Mean Cross-Validation R^2 Score: {sum(test_r2s) / len(test_r2s):.10f}")
        # print("\nTest Set (Final Evaluation):")
        # print(f"Final Test RMSE: {final_test_rmse:.10f}")
        # print(f"Final Test R^2 Score: {final_test_r2:.10f}")
        # print("\n\n")
    def run(self):
        print("Processing lyrics!\n")

        print("Loading dataset!\n")
        # Read input DataFrame
        song_features_df = pd.read_excel(utils.get_output_directory_path() + '/song_features_output.xlsx')

        original_df = pd.read_excel(utils.get_output_directory_path() + '/skladba_dataframe_cleaned.xlsx')

        # Get columns for audio
        audio_columns = [
            col for col in original_df.columns
            if any(keyword in col.lower() for keyword in ['songs', 'rhythm', 'melody', 'harmony'])
        ]

        audio_columns_df = original_df[audio_columns]
        #merged = utils.merge_dataframes(song_features_df, audio_columns_df, "merged_dataframe_songs.xlsx", "songs",
        #"inner")

        columns_to_predict = ['rhythm_mean', 'harmony_mean', 'melody_mean']

        # Models that are being used in ML algorithm
        models = [
            RandomForestRegressor(n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1,
                                  random_state=42),
            GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, min_samples_split=2,
                                      min_samples_leaf=1, random_state=42),
            ExtraTreesRegressor(n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1,
                                random_state=42),
            DecisionTreeRegressor(max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=42),
            LinearRegression(),
            SVR(kernel='rbf'),
            # SGDRegressor(alpha=0.0001, penalty='l2', loss='squared_loss', random_state=42) currently not working

        ]

        # classifiers = [
        #     SVC(kernel='linear'),
        #     RandomForestClassifier(n_estimators=100, max_depth=None),
        #     LogisticRegression(C=1.0, penalty='l2', random_state=42),
        #     RidgeClassifier(alpha=1.0, random_state=42),
        #     SGDClassifier(alpha=0.0001, penalty='l2', loss='log', random_state=42),
        #     DecisionTreeClassifier(max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=42),
        #     KNeighborsClassifier(n_neighbors=10)  # Not sure for parameters
        #     # NaiveBayesClassifier() How to implement this and fix parameters.
        # ]
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
            }
        ]

        # print("Hyperparameters tuning!")
        # for model, param_grid in zip(models, param_grids):
        #    for column in columns_to_predict:
        #        self.find_optimal_hyperparameters(merged, column, model, param_grid)


        print("Evaluating Regressors!\n")
        for model in models:
            for column in columns_to_predict:
                self.evaluation(audio_columns_df, song_features_df, column, model)

        # print("Evaluating Classifiers!\n")
        # for model in models:
        #     for column in columns_to_predict:
        #         self.evaluation_classifier(merged, column, model)
