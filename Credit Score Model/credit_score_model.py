import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import joblib
import logging
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CreditScoreModel:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = None
        self.scaler = None
        self.imputer = None
        self.feature_names = None

    def load_data(self, filepath):
        """
        Load data from a CSV file.

        Args:
            filepath (str): Path to the CSV file.

        Returns:
            pandas.DataFrame: Loaded data.
        """
        try:
            data = pd.read_csv(filepath)
            logging.info(f"Data loaded successfully from {filepath}")
            return data
        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")
            raise

    def preprocess_data(self, data, target_column):
        """
        Preprocess the data by splitting features and target, and handling missing values.

        Args:
            data (pandas.DataFrame): Input data.
            target_column (str): Name of the target column.

        Returns:
            tuple: X (features) and y (target) dataframes.
        """
        try:
            X = data.drop(target_column, axis=1)
            y = data[target_column]
            self.feature_names = X.columns.tolist()

            # Handle missing values
            self.imputer = SimpleImputer(strategy='median')
            X = pd.DataFrame(self.imputer.fit_transform(X), columns=X.columns)

            logging.info("Data preprocessing completed")
            return X, y
        except Exception as e:
            logging.error(f"Error in data preprocessing: {str(e)}")
            raise

    def split_data(self, X, y, test_size=0.2):
        """
        Split the data into training and testing sets.

        Args:
            X (pandas.DataFrame): Features.
            y (pandas.Series): Target variable.
            test_size (float): Proportion of the dataset to include in the test split.

        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=self.random_state)
            logging.info(f"Data split into train and test sets. Train set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error(f"Error in splitting data: {str(e)}")
            raise

    def create_pipeline(self):
        """
        Create a scikit-learn pipeline with preprocessing and model steps.

        Returns:
            sklearn.pipeline.Pipeline: Model pipeline.
        """
        try:
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', RandomForestClassifier(random_state=self.random_state))
            ])
            logging.info("Model pipeline created")
            return pipeline
        except Exception as e:
            logging.error(f"Error in creating pipeline: {str(e)}")
            raise

    def train_model(self, X_train, y_train):
        """
        Train the model using GridSearchCV for hyperparameter tuning.

        Args:
            X_train (pandas.DataFrame): Training features.
            y_train (pandas.Series): Training target.
        """
        try:
            pipeline = self.create_pipeline()
            param_grid = {
                'classifier__n_estimators': [100, 200, 300],
                'classifier__max_depth': [None, 10, 20, 30],
                'classifier__min_samples_split': [2, 5, 10],
                'classifier__min_samples_leaf': [1, 2, 4]
            }
            self.model = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=1)
            self.model.fit(X_train, y_train)
            logging.info("Model training completed")
            logging.info(f"Best parameters: {self.model.best_params_}")
        except Exception as e:
            logging.error(f"Error in model training: {str(e)}")
            raise

    def evaluate_model(self, X_test, y_test):
        """
        Evaluate the trained model on the test set.

        Args:
            X_test (pandas.DataFrame): Test features.
            y_test (pandas.Series): Test target.

        Returns:
            dict: Dictionary containing evaluation metrics.
        """
        try:
            y_pred = self.model.predict(X_test)
            y_pred_proba = self.model.predict_proba(X_test)[:, 1]
            
            report = classification_report(y_test, y_pred, output_dict=True)
            confusion_mat = confusion_matrix(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            
            evaluation = {
                'classification_report': report,
                'confusion_matrix': confusion_mat,
                'roc_auc_score': roc_auc
            }
            
            logging.info("Model evaluation completed")
            logging.info(f"ROC AUC Score: {roc_auc}")
            return evaluation
        except Exception as e:
            logging.error(f"Error in model evaluation: {str(e)}")
            raise

    def plot_feature_importance(self, top_n=10):
        """
        Plot feature importance of the trained model.

        Args:
            top_n (int): Number of top features to display.
        """
        try:
            feature_importance = self.model.best_estimator_.named_steps['classifier'].feature_importances_
            feature_importance_df = pd.DataFrame({'feature': self.feature_names, 'importance': feature_importance})
            feature_importance_df = feature_importance_df.sort_values('importance', ascending=False).head(top_n)

            plt.figure(figsize=(10, 6))
            sns.barplot(x='importance', y='feature', data=feature_importance_df)
            plt.title(f'Top {top_n} Feature Importance')
            plt.tight_layout()
            plt.savefig('feature_importance.png')
            logging.info("Feature importance plot saved as 'feature_importance.png'")
        except Exception as e:
            logging.error(f"Error in plotting feature importance: {str(e)}")
            raise

    def save_model(self, filepath):
        """
        Save the trained model to a file.

        Args:
            filepath (str): Path to save the model.
        """
        try:
            joblib.dump(self.model, filepath)
            logging.info(f"Model saved to {filepath}")
        except Exception as e:
            logging.error(f"Error in saving model: {str(e)}")
            raise

    def load_model(self, filepath):
        """
        Load a trained model from a file.

        Args:
            filepath (str): Path to the saved model.
        """
        try:
            self.model = joblib.load(filepath)
            logging.info(f"Model loaded from {filepath}")
        except Exception as e:
            logging.error(f"Error in loading model: {str(e)}")
            raise

def main():
    # Initialize the model
    credit_model = CreditScoreModel()

    # Load and preprocess data
    data = credit_model.load_data('credit_data.csv')
    X, y = credit_model.preprocess_data(data, target_column='credit_score')

    # Split the data
    X_train, X_test, y_train, y_test = credit_model.split_data(X, y)

    # Train the model
    credit_model.train_model(X_train, y_train)

    # Evaluate the model
    evaluation = credit_model.evaluate_model(X_test, y_test)
    print("Classification Report:")
    print(pd.DataFrame(evaluation['classification_report']).transpose())
    print("\nConfusion Matrix:")
    print(evaluation['confusion_matrix'])
    print(f"\nROC AUC Score: {evaluation['roc_auc_score']}")

    # Plot feature importance
    credit_model.plot_feature_importance()

    # Save the model
    credit_model.save_model('credit_score_model.joblib')

if __name__ == "__main__":
    main()

