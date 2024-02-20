# import necessary libraries
import os
import joblib
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score

import warnings
warnings.filterwarnings("ignore")

# Function to preprocess the data
# Function to preprocess the data
def preprocess(input_data, target, numeric_features, categorical_features, train_features_output_path, train_labels_output_path, val_features_output_path, val_labels_output_path, test_features_output_path, test_labels_output_path):
    """
    This function preprocesses the data and saves the train, validation and test sets to csv files.
    """
    # read the data
    AMR_Data = input_data

    # split the data into train and test sets
    Xtrain, Xtest, ytrain, ytest = train_test_split(AMR_Data.drop(columns=[target]), AMR_Data[target],
                                                    test_size=0.2,
                                                    random_state=42)

    # split the train data into train and validation sets
    Xtrain, Xval, ytrain, yval = train_test_split(Xtrain, ytrain,
                                                  test_size=0.2,
                                                  random_state=42)

    # create a preprocessor object to preprocess the data
    preprocessor = make_column_transformer(
        (StandardScaler(), numeric_features),
        (OneHotEncoder(handle_unknown='ignore',
                       sparse=False), categorical_features)
    )

    transformed_Xtrain = preprocessor.fit_transform(Xtrain)
    transformed_Xval = preprocessor.transform(Xval)
    transformed_Xtest = preprocessor.transform(Xtest)

    # save the validation set to csv file
    pd.DataFrame(transformed_Xval).to_csv(val_features_output_path,
                                          header=False, index=False)
    pd.DataFrame(transformed_Xtrain).to_csv(train_features_output_path,
                                            header=False, index=False)
    pd.DataFrame(transformed_Xtest).to_csv(test_features_output_path,
                                           header=False, index=False)

    ytrain.to_csv(train_labels_output_path, header=False, index=False)
    yval.to_csv(val_labels_output_path, header=False, index=False)
    ytest.to_csv(test_labels_output_path, header=False, index=False)

# Function to build a Logistic Regression model
from sklearn.linear_model import LogisticRegression
# metrics for evaluating the logistic regression model performance
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report, confusion_matrix

def training_lr(train_features_data, train_labels_data, val_features_data, val_labels_data, model_output_directory):
    """
    This function trains the model and saves the model to the model_output_directory.
    """
    X_train = pd.read_csv(train_features_data, header=None)
    y_train = pd.read_csv(train_labels_data, header=None)

    model_lr = LogisticRegression()

    model_lr.fit(X_train, y_train)

    X_val = pd.read_csv(val_features_data, header=None)
    y_val = pd.read_csv(val_labels_data, header=None)

    y_pred_val = model_lr.predict(X_val)

    print("Validation Set Metrics")
    print(f"Accuracy: {accuracy_score(y_val, y_pred_val)}")
    print(f"Precision: {precision_score(y_val, y_pred_val)}")
    print(f"Recall: {recall_score(y_val, y_pred_val)}")
    print(f"F1 Score: {f1_score(y_val, y_pred_val)}")
    print(f"ROC AUC: {roc_auc_score(y_val, y_pred_val)}")
    print(f"Confusion Matrix: \n{confusion_matrix(y_val, y_pred_val)}")
    print(f"Classification Report: \n{classification_report(y_val, y_pred_val)}")

    # print accuracy score on the training and validation sets
    print(f"Training Accuracy: {model_lr.score(X_train, y_train)}")
    print(f"Validation Accuracy: {model_lr.score(X_val, y_val)}")

    print(f"Saving model to {model_output_directory}")
    joblib.dump(model_lr, model_output_directory)

# Function to evaluate the Logistic Regression model

def evaluation_lr(model_path, test_features_data, test_labels_data, evaluation_output_path):
    """
    This function evaluates the model and saves the evaluation report to the evaluation_output_path.
    """
    model = joblib.load(model_path)

    X_test = pd.read_csv(test_features_data, header=None)
    y_test = pd.read_csv(test_labels_data, header=None)

    y_pred = model.predict(X_test)

    print("Test Set Metrics")
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(f"Precision: {precision_score(y_test, y_pred)}")
    print(f"Recall: {recall_score(y_test, y_pred)}")
    print(f"F1 Score: {f1_score(y_test, y_pred)}")
    print(f"ROC AUC: {roc_auc_score(y_test, y_pred)}")
    print(f"Confusion Matrix: \n{confusion_matrix(y_test, y_pred)}")
    print(f"Classification Report: \n{classification_report(y_test, y_pred)}")

    report_dict = {
        "classification_metrics": {
            "accuracy": {
                "value": accuracy_score(y_test, y_pred)
            },
            "precision": {
                "value": precision_score(y_test, y_pred)
            },
            "recall": {
                "value": recall_score(y_test, y_pred)
            },
            "f1": {
                "value": f1_score(y_test, y_pred)
            },
            "roc_auc": {
                "value": roc_auc_score(y_test, y_pred)
            }
        }
    }

    with open(evaluation_output_path, "w") as f:
          f.write(json.dumps(report_dict))
    return report_dict, y_pred

# Function to build a Decision Tree Classifier model
from sklearn.tree import DecisionTreeClassifier
# metrics for evaluating the decision tree classifier model performance
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report, confusion_matrix

def training_dtc(train_features_data, train_labels_data, val_features_data, val_labels_data, model_output_directory):
        """
        This function trains the model and saves the model to the model_output_directory.
        """
        X_train = pd.read_csv(train_features_data, header=None)
        y_train = pd.read_csv(train_labels_data, header=None)
        
        model_dtc = DecisionTreeClassifier()
        
        model_dtc.fit(X_train, y_train)
        
        X_val = pd.read_csv(val_features_data, header=None)
        y_val = pd.read_csv(val_labels_data, header=None)
        
        y_pred_val = model_dtc.predict(X_val)
        
        print("Validation Set Metrics")
        print(f"Accuracy: {accuracy_score(y_val, y_pred_val)}")
        print(f"Precision: {precision_score(y_val, y_pred_val)}")
        print(f"Recall: {recall_score(y_val, y_pred_val)}")
        print(f"F1 Score: {f1_score(y_val, y_pred_val)}")
        print(f"ROC AUC: {roc_auc_score(y_val, y_pred_val)}")
        print(f"Confusion Matrix: \n{confusion_matrix(y_val, y_pred_val)}")
        print(f"Classification Report: \n{classification_report(y_val, y_pred_val)}")
        
        # print accuracy score on the training and validation sets
        print(f"Training Accuracy: {model_dtc.score(X_train, y_train)}")
        print(f"Validation Accuracy: {model_dtc.score(X_val, y_val)}")
        
        print(f"Saving model to {model_output_directory}")
        joblib.dump(model_dtc, model_output_directory)

# Function to evaluate the Decision Tree Classifier model
def evaluation_dtc(model_path, test_features_data, test_labels_data, evaluation_output_path):
        """
        This function evaluates the model and saves the evaluation report to the evaluation_output_path.
        """
        model = joblib.load(model_path)
        
        X_test = pd.read_csv(test_features_data, header=None)
        y_test = pd.read_csv(test_labels_data, header=None)
        
        y_pred = model.predict(X_test)
        
        print("Test Set Metrics")
        print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
        print(f"Precision: {precision_score(y_test, y_pred)}")
        print(f"Recall: {recall_score(y_test, y_pred)}")
        print(f"F1 Score: {f1_score(y_test, y_pred)}")
        print(f"ROC AUC: {roc_auc_score(y_test, y_pred)}")
        print(f"Confusion Matrix: \n{confusion_matrix(y_test, y_pred)}")
        print(f"Classification Report: \n{classification_report(y_test, y_pred)}")
        
        report_dict = {
            "classification_metrics": {
                "accuracy": {
                    "value": accuracy_score(y_test, y_pred)
                },
                "precision": {
                    "value": precision_score(y_test, y_pred)
                },
                "recall": {
                    "value": recall_score(y_test, y_pred)
                },
                "f1": {
                    "value": f1_score(y_test, y_pred)
                },
                "roc_auc": {
                    "value": roc_auc_score(y_test, y_pred)
                }
            }
        }
        
        with open(evaluation_output_path, "w") as f:
            f.write(json.dumps(report_dict))
            

# Function to build a k-nearest neighbors classifier model
from sklearn.neighbors import KNeighborsClassifier
# metrics for evaluating the k-nearest neighbors classifier model performance
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report, confusion_matrix

def training_knn(train_features_data, train_labels_data, val_features_data, val_labels_data, model_output_directory):
        """
        This function trains the model and saves the model to the model_output_directory.
        """
        X_train = pd.read_csv(train_features_data, header=None)
        y_train = pd.read_csv(train_labels_data, header=None)
        
        model_knn = KNeighborsClassifier()
        
        model_knn.fit(X_train, y_train)
        
        X_val = pd.read_csv(val_features_data, header=None)
        y_val = pd.read_csv(val_labels_data, header=None)
        
        y_pred_val = model_knn.predict(X_val)
        
        print("Validation Set Metrics")
        print(f"Accuracy: {accuracy_score(y_val, y_pred_val)}")
        print(f"Precision: {precision_score(y_val, y_pred_val)}")
        print(f"Recall: {recall_score(y_val, y_pred_val)}")
        print(f"F1 Score: {f1_score(y_val, y_pred_val)}")
        print(f"ROC AUC: {roc_auc_score(y_val, y_pred_val)}")
        print(f"Confusion Matrix: \n{confusion_matrix(y_val, y_pred_val)}")
        print(f"Classification Report: \n{classification_report(y_val, y_pred_val)}")
        
        # print accuracy score on the training and validation sets
        print(f"Training Accuracy: {model_knn.score(X_train, y_train)}")
        print(f"Validation Accuracy: {model_knn.score(X_val, y_val)}")
        
        print(f"Saving model to {model_output_directory}")
        joblib.dump(model_knn, model_output_directory)

# Function to evaluate the k-nearest neighbors classifier model
def evaluation_knn(model_path, test_features_data, test_labels_data, evaluation_output_path):
        """
        This function evaluates the model and saves the evaluation report to the evaluation_output_path.
        """
        model = joblib.load(model_path)
        
        X_test = pd.read_csv(test_features_data, header=None)
        y_test = pd.read_csv(test_labels_data, header=None)
        
        y_pred = model.predict(X_test)
        
        print("Test Set Metrics")
        print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
        print(f"Precision: {precision_score(y_test, y_pred)}")
        print(f"Recall: {recall_score(y_test, y_pred)}")
        print(f"F1 Score: {f1_score(y_test, y_pred)}")
        print(f"ROC AUC: {roc_auc_score(y_test, y_pred)}")
        print(f"Confusion Matrix: \n{confusion_matrix(y_test, y_pred)}")
        print(f"Classification Report: \n{classification_report(y_test, y_pred)}")
        
        report_dict = {
            "classification_metrics": {
                "accuracy": {
                    "value": accuracy_score(y_test, y_pred)
                },
                "precision": {
                    "value": precision_score(y_test, y_pred)
                },
                "recall": {
                    "value": recall_score(y_test, y_pred)
                },
                "f1": {
                    "value": f1_score(y_test, y_pred)
                },
                "roc_auc": {
                    "value": roc_auc_score(y_test, y_pred)
                }
            }
        }
        
        with open(evaluation_output_path, "w") as f:
            f.write(json.dumps(report_dict))


# Function to build a Random Forest Classifier model
from sklearn.ensemble import RandomForestClassifier
# metrics for evaluating the random forest classifier model performance
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report, confusion_matrix

def training_rfc(train_features_data, train_labels_data, val_features_data, val_labels_data, model_output_directory):
        """
        This function trains the model and saves the model to the model_output_directory.
        """
        X_train = pd.read_csv(train_features_data, header=None)
        y_train = pd.read_csv(train_labels_data, header=None)
        
        model_rfc = RandomForestClassifier()
        
        model_rfc.fit(X_train, y_train)
        
        X_val = pd.read_csv(val_features_data, header=None)
        y_val = pd.read_csv(val_labels_data, header=None)
        
        y_pred_val = model_rfc.predict(X_val)
        
        print("Validation Set Metrics")
        print(f"Accuracy: {accuracy_score(y_val, y_pred_val)}")
        print(f"Precision: {precision_score(y_val, y_pred_val)}")
        print(f"Recall: {recall_score(y_val, y_pred_val)}")
        print(f"F1 Score: {f1_score(y_val, y_pred_val)}")
        print(f"ROC AUC: {roc_auc_score(y_val, y_pred_val)}")
        print(f"Confusion Matrix: \n{confusion_matrix(y_val, y_pred_val)}")
        print(f"Classification Report: \n{classification_report(y_val, y_pred_val)}")
        
        # print accuracy score on the training and validation sets
        print(f"Training Accuracy: {model_rfc.score(X_train, y_train)}")
        print(f"Validation Accuracy: {model_rfc.score(X_val, y_val)}")
        
        print(f"Saving model to {model_output_directory}")
        joblib.dump(model_rfc, model_output_directory)

# Function to evaluate the Random Forest Classifier model
def evaluation_rfc(model_path, test_features_data, test_labels_data, evaluation_output_path):
        """
        This function evaluates the model and saves the evaluation report to the evaluation_output_path.
        """
        model = joblib.load(model_path)
        
        X_test = pd.read_csv(test_features_data, header=None)
        y_test = pd.read_csv(test_labels_data, header=None)
        
        y_pred = model.predict(X_test)
        
        print("Test Set Metrics")
        print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
        print(f"Precision: {precision_score(y_test, y_pred)}")
        print(f"Recall: {recall_score(y_test, y_pred)}")
        print(f"F1 Score: {f1_score(y_test, y_pred)}")
        print(f"ROC AUC: {roc_auc_score(y_test, y_pred)}")
        print(f"Confusion Matrix: \n{confusion_matrix(y_test, y_pred)}")
        print(f"Classification Report: \n{classification_report(y_test, y_pred)}")
        
        report_dict = {
            "classification_metrics": {
                "accuracy": {
                    "value": accuracy_score(y_test, y_pred)
                },
                "precision": {
                    "value": precision_score(y_test, y_pred)
                },
                "recall": {
                    "value": recall_score(y_test, y_pred)
                },
                "f1": {
                    "value": f1_score(y_test, y_pred)
                },
                "roc_auc": {
                    "value": roc_auc_score(y_test, y_pred)
                }
            }
        }
        
        with open(evaluation_output_path, "w") as f:
            f.write(json.dumps(report_dict))

# Function to build a Gradient Boosting Classifier model
from sklearn.ensemble import GradientBoostingClassifier
# metrics for evaluating the gradient boosting classifier model performance
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report, confusion_matrix

def training_gbc(train_features_data, train_labels_data, val_features_data, val_labels_data, model_output_directory):
        """
        This function trains the model and saves the model to the model_output_directory.
        """
        X_train = pd.read_csv(train_features_data, header=None)
        y_train = pd.read_csv(train_labels_data, header=None)
        
        model_gbc = GradientBoostingClassifier()
        
        model_gbc.fit(X_train, y_train)
        
        X_val = pd.read_csv(val_features_data, header=None)
        y_val = pd.read_csv(val_labels_data, header=None)
        
        y_pred_val = model_gbc.predict(X_val)
        
        print("Validation Set Metrics")
        print(f"Accuracy: {accuracy_score(y_val, y_pred_val)}")
        print(f"Precision: {precision_score(y_val, y_pred_val)}")
        print(f"Recall: {recall_score(y_val, y_pred_val)}")
        print(f"F1 Score: {f1_score(y_val, y_pred_val)}")
        print(f"ROC AUC: {roc_auc_score(y_val, y_pred_val)}")
        print(f"Confusion Matrix: \n{confusion_matrix(y_val, y_pred_val)}")
        print(f"Classification Report: \n{classification_report(y_val, y_pred_val)}")
        
        # print accuracy score on the training and validation sets
        print(f"Training Accuracy: {model_gbc.score(X_train, y_train)}")
        print(f"Validation Accuracy: {model_gbc.score(X_val, y_val)}")
        
        print(f"Saving model to {model_output_directory}")
        joblib.dump(model_gbc, model_output_directory)

# Function to evaluate the Gradient Boosting Classifier model
def evaluation_gbc(model_path, test_features_data, test_labels_data, evaluation_output_path):
        """
        This function evaluates the model and saves the evaluation report to the evaluation_output_path.
        """
        model = joblib.load(model_path)
        
        X_test = pd.read_csv(test_features_data, header=None)
        y_test = pd.read_csv(test_labels_data, header=None)
        
        y_pred = model.predict(X_test)
        
        print("Test Set Metrics")
        print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
        print(f"Precision: {precision_score(y_test, y_pred)}")
        print(f"Recall: {recall_score(y_test, y_pred)}")
        print(f"F1 Score: {f1_score(y_test, y_pred)}")
        print(f"ROC AUC: {roc_auc_score(y_test, y_pred)}")
        print(f"Confusion Matrix: \n{confusion_matrix(y_test, y_pred)}")
        print(f"Classification Report: \n{classification_report(y_test, y_pred)}")
        
        report_dict = {
            "classification_metrics": {
                "accuracy": {
                    "value": accuracy_score(y_test, y_pred)
                },
                "precision": {
                    "value": precision_score(y_test, y_pred)
                },
                "recall": {
                    "value": recall_score(y_test, y_pred)
                },
                "f1": {
                    "value": f1_score(y_test, y_pred)
                },
                "roc_auc": {
                    "value": roc_auc_score(y_test, y_pred)
                }
            }
        }
        
        with open(evaluation_output_path, "w") as f:
            f.write(json.dumps(report_dict))

# Function to build a Bagging Classifier model
from sklearn.ensemble import BaggingClassifier
# metrics for evaluating the bagging classifier model performance
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report, confusion_matrix

def training_bc(train_features_data, train_labels_data, val_features_data, val_labels_data, model_output_directory):
        """
        This function trains the model and saves the model to the model_output_directory.
        """
        X_train = pd.read_csv(train_features_data, header=None)
        y_train = pd.read_csv(train_labels_data, header=None)
        
        model_bc = BaggingClassifier()
        
        model_bc.fit(X_train, y_train)
        
        X_val = pd.read_csv(val_features_data, header=None)
        y_val = pd.read_csv(val_labels_data, header=None)
        
        y_pred_val = model_bc.predict(X_val)
        
        print("Validation Set Metrics")
        print(f"Accuracy: {accuracy_score(y_val, y_pred_val)}")
        print(f"Precision: {precision_score(y_val, y_pred_val)}")
        print(f"Recall: {recall_score(y_val, y_pred_val)}")
        print(f"F1 Score: {f1_score(y_val, y_pred_val)}")
        print(f"ROC AUC: {roc_auc_score(y_val, y_pred_val)}")
        print(f"Confusion Matrix: \n{confusion_matrix(y_val, y_pred_val)}")
        print(f"Classification Report: \n{classification_report(y_val, y_pred_val)}")
        
        # print accuracy score on the training and validation sets
        print(f"Training Accuracy: {model_bc.score(X_train, y_train)}")
        print(f"Validation Accuracy: {model_bc.score(X_val, y_val)}")
        
        print(f"Saving model to {model_output_directory}")
        joblib.dump(model_bc, model_output_directory)

# Function to evaluate the Bagging Classifier model
def evaluation_bc(model_path, test_features_data, test_labels_data, evaluation_output_path):
        """
        This function evaluates the model and saves the evaluation report to the evaluation_output_path.
        """
        model = joblib.load(model_path)
        
        X_test = pd.read_csv(test_features_data, header=None)
        y_test = pd.read_csv(test_labels_data, header=None)
        
        y_pred = model.predict(X_test)
        
        print("Test Set Metrics")
        print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
        print(f"Precision: {precision_score(y_test, y_pred)}")
        print(f"Recall: {recall_score(y_test, y_pred)}")
        print(f"F1 Score: {f1_score(y_test, y_pred)}")
        print(f"ROC AUC: {roc_auc_score(y_test, y_pred)}")
        print(f"Confusion Matrix: \n{confusion_matrix(y_test, y_pred)}")
        print(f"Classification Report: \n{classification_report(y_test, y_pred)}")
        
        report_dict = {
            "classification_metrics": {
                "accuracy": {
                    "value": accuracy_score(y_test, y_pred)
                },
                "precision": {
                    "value": precision_score(y_test, y_pred)
                },
                "recall": {
                    "value": recall_score(y_test, y_pred)
                },
                "f1": {
                    "value": f1_score(y_test, y_pred)
                },
                "roc_auc": {
                    "value": roc_auc_score(y_test, y_pred)
                }
            }
        }
        
        with open(evaluation_output_path, "w") as f:
            f.write(json.dumps(report_dict))

# Function to build a Ada Boost Classifier model
from sklearn.ensemble import AdaBoostClassifier
# metrics for evaluating the ada boost classifier model performance
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report, confusion_matrix

def training_abc(train_features_data, train_labels_data, val_features_data, val_labels_data, model_output_directory):
        """
        This function trains the model and saves the model to the model_output_directory.
        """
        X_train = pd.read_csv(train_features_data, header=None)
        y_train = pd.read_csv(train_labels_data, header=None)
        
        model_abc = AdaBoostClassifier()
        
        model_abc.fit(X_train, y_train)
        
        X_val = pd.read_csv(val_features_data, header=None)
        y_val = pd.read_csv(val_labels_data, header=None)
        
        y_pred_val = model_abc.predict(X_val)
        
        print("Validation Set Metrics")
        print(f"Accuracy: {accuracy_score(y_val, y_pred_val)}")
        print(f"Precision: {precision_score(y_val, y_pred_val)}")
        print(f"Recall: {recall_score(y_val, y_pred_val)}")
        print(f"F1 Score: {f1_score(y_val, y_pred_val)}")
        print(f"ROC AUC: {roc_auc_score(y_val, y_pred_val)}")
        print(f"Confusion Matrix: \n{confusion_matrix(y_val, y_pred_val)}")
        print(f"Classification Report: \n{classification_report(y_val, y_pred_val)}")
        
        # print accuracy score on the training and validation sets
        print(f"Training Accuracy: {model_abc.score(X_train, y_train)}")
        print(f"Validation Accuracy: {model_abc.score(X_val, y_val)}")
        
        print(f"Saving model to {model_output_directory}")
        joblib.dump(model_abc, model_output_directory)

# Function to evaluate the Ada Boost Classifier model
def evaluation_abc(model_path, test_features_data, test_labels_data, evaluation_output_path):
        """
        This function evaluates the model and saves the evaluation report to the evaluation_output_path.
        """
        model = joblib.load(model_path)
        
        X_test = pd.read_csv(test_features_data, header=None)
        y_test = pd.read_csv(test_labels_data, header=None)
        
        y_pred = model.predict(X_test)
        
        print("Test Set Metrics")
        print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
        print(f"Precision: {precision_score(y_test, y_pred)}")
        print(f"Recall: {recall_score(y_test, y_pred)}")
        print(f"F1 Score: {f1_score(y_test, y_pred)}")
        print(f"ROC AUC: {roc_auc_score(y_test, y_pred)}")
        print(f"Confusion Matrix: \n{confusion_matrix(y_test, y_pred)}")
        print(f"Classification Report: \n{classification_report(y_test, y_pred)}")
        
        report_dict = {
            "classification_metrics": {
                "accuracy": {
                    "value": accuracy_score(y_test, y_pred)
                },
                "precision": {
                    "value": precision_score(y_test, y_pred)
                },
                "recall": {
                    "value": recall_score(y_test, y_pred)
                },
                "f1": {
                    "value": f1_score(y_test, y_pred)
                },
                "roc_auc": {
                    "value": roc_auc_score(y_test, y_pred)
                }
            }
        }
        
        with open(evaluation_output_path, "w") as f:
            f.write(json.dumps(report_dict))

# Function to build a XGBoost Classifier model
from xgboost import XGBClassifier
# metrics for evaluating the xgboost classifier model performance
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report, confusion_matrix

def training_xgbc(train_features_data, train_labels_data, val_features_data, val_labels_data, model_output_directory):
        """
        This function trains the model and saves the model to the model_output_directory.
        """
        X_train = pd.read_csv(train_features_data, header=None)
        y_train = pd.read_csv(train_labels_data, header=None)
        
        model_xgbc = XGBClassifier()
        
        model_xgbc.fit(X_train, y_train)
        
        X_val = pd.read_csv(val_features_data, header=None)
        y_val = pd.read_csv(val_labels_data, header=None)
        
        y_pred_val = model_xgbc.predict(X_val)
        
        print("Validation Set Metrics")
        print(f"Accuracy: {accuracy_score(y_val, y_pred_val)}")
        print(f"Precision: {precision_score(y_val, y_pred_val)}")
        print(f"Recall: {recall_score(y_val, y_pred_val)}")
        print(f"F1 Score: {f1_score(y_val, y_pred_val)}")
        print(f"ROC AUC: {roc_auc_score(y_val, y_pred_val)}")
        print(f"Confusion Matrix: \n{confusion_matrix(y_val, y_pred_val)}")
        print(f"Classification Report: \n{classification_report(y_val, y_pred_val)}")
        
        # print accuracy score on the training and validation sets
        print(f"Training Accuracy: {model_xgbc.score(X_train, y_train)}")
        print(f"Validation Accuracy: {model_xgbc.score(X_val, y_val)}")
        
        print(f"Saving model to {model_output_directory}")
        joblib.dump(model_xgbc, model_output_directory)

# Function to evaluate the XGBoost Classifier model
def evaluation_xgbc(model_path, test_features_data, test_labels_data, evaluation_output_path):
        """
        This function evaluates the model and saves the evaluation report to the evaluation_output_path.
        """
        model = joblib.load(model_path)
        
        X_test = pd.read_csv(test_features_data, header=None)
        y_test = pd.read_csv(test_labels_data, header=None)
        
        y_pred = model.predict(X_test)
        
        print("Test Set Metrics")
        print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
        print(f"Precision: {precision_score(y_test, y_pred)}")
        print(f"Recall: {recall_score(y_test, y_pred)}")
        print(f"F1 Score: {f1_score(y_test, y_pred)}")
        print(f"ROC AUC: {roc_auc_score(y_test, y_pred)}")
        print(f"Confusion Matrix: \n{confusion_matrix(y_test, y_pred)}")
        print(f"Classification Report: \n{classification_report(y_test, y_pred)}")
        
        report_dict = {
            "classification_metrics": {
                "accuracy": {
                    "value": accuracy_score(y_test, y_pred)
                },
                "precision": {
                    "value": precision_score(y_test, y_pred)
                },
                "recall": {
                    "value": recall_score(y_test, y_pred)
                },
                "f1": {
                    "value": f1_score(y_test, y_pred)
                },
                "roc_auc": {
                    "value": roc_auc_score(y_test, y_pred)
                }
            }
        }
        
        with open(evaluation_output_path, "w") as f:
            f.write(json.dumps(report_dict))

# Function to build a support vector classifier model
from sklearn.svm import SVC
# metrics for evaluating the support vector classifier model performance
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report, confusion_matrix

def training_svc(train_features_data, train_labels_data, val_features_data, val_labels_data, model_output_directory):
        """
        This function trains the model and saves the model to the model_output_directory.
        """
        X_train = pd.read_csv(train_features_data, header=None)
        y_train = pd.read_csv(train_labels_data, header=None)
        
        model_svc = SVC()
        
        model_svc.fit(X_train, y_train)
        
        X_val = pd.read_csv(val_features_data, header=None)
        y_val = pd.read_csv(val_labels_data, header=None)
        
        y_pred_val = model_svc.predict(X_val)
        
        print("Validation Set Metrics")
        print(f"Accuracy: {accuracy_score(y_val, y_pred_val)}")
        print(f"Precision: {precision_score(y_val, y_pred_val)}")
        print(f"Recall: {recall_score(y_val, y_pred_val)}")
        print(f"F1 Score: {f1_score(y_val, y_pred_val)}")
        print(f"ROC AUC: {roc_auc_score(y_val, y_pred_val)}")
        print(f"Confusion Matrix: \n{confusion_matrix(y_val, y_pred_val)}")
        print(f"Classification Report: \n{classification_report(y_val, y_pred_val)}")
        
        # print accuracy score on the training and validation sets
        print(f"Training Accuracy: {model_svc.score(X_train, y_train)}")
        print(f"Validation Accuracy: {model_svc.score(X_val, y_val)}")
        
        print(f"Saving model to {model_output_directory}")
        joblib.dump(model_svc, model_output_directory)

# Function to evaluate the Support Vector Classifier model
def evaluation_svc(model_path, test_features_data, test_labels_data, evaluation_output_path):
        """
        This function evaluates the model and saves the evaluation report to the evaluation_output_path.
        """
        model = joblib.load(model_path)
        
        X_test = pd.read_csv(test_features_data, header=None)
        y_test = pd.read_csv(test_labels_data, header=None)
        
        y_pred = model.predict(X_test)
        
        print("Test Set Metrics")
        print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
        print(f"Precision: {precision_score(y_test, y_pred)}")
        print(f"Recall: {recall_score(y_test, y_pred)}")
        print(f"F1 Score: {f1_score(y_test, y_pred)}")
        print(f"ROC AUC: {roc_auc_score(y_test, y_pred)}")
        print(f"Confusion Matrix: \n{confusion_matrix(y_test, y_pred)}")
        print(f"Classification Report: \n{classification_report(y_test, y_pred)}")
        
        report_dict = {
            "classification_metrics": {
                "accuracy": {
                    "value": accuracy_score(y_test, y_pred)
                },
                "precision": {
                    "value": precision_score(y_test, y_pred)
                },
                "recall": {
                    "value": recall_score(y_test, y_pred)
                },
                "f1": {
                    "value": f1_score(y_test, y_pred)
                },
                "roc_auc": {
                    "value": roc_auc_score(y_test, y_pred)
                }
            }
        }
        
        with open(evaluation_output_path, "w") as f:
            f.write(json.dumps(report_dict))

# Function to build a Naive Bayes Classifier model
from sklearn.naive_bayes import GaussianNB
# metrics for evaluating the naive bayes classifier model performance
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report, confusion_matrix

def training_nb(train_features_data, train_labels_data, val_features_data, val_labels_data, model_output_directory):
        """
        This function trains the model and saves the model to the model_output_directory.
        """
        X_train = pd.read_csv(train_features_data, header=None)
        y_train = pd.read_csv(train_labels_data, header=None)
        
        model_nb = GaussianNB()
        
        model_nb.fit(X_train, y_train)
        
        X_val = pd.read_csv(val_features_data, header=None)
        y_val = pd.read_csv(val_labels_data, header=None)
        
        y_pred_val = model_nb.predict(X_val)
        
        print("Validation Set Metrics")
        print(f"Accuracy: {accuracy_score(y_val, y_pred_val)}")
        print(f"Precision: {precision_score(y_val, y_pred_val)}")
        print(f"Recall: {recall_score(y_val, y_pred_val)}")
        print(f"F1 Score: {f1_score(y_val, y_pred_val)}")
        print(f"ROC AUC: {roc_auc_score(y_val, y_pred_val)}")
        print(f"Confusion Matrix: \n{confusion_matrix(y_val, y_pred_val)}")
        print(f"Classification Report: \n{classification_report(y_val, y_pred_val)}")
        
        # print accuracy score on the training and validation sets
        print(f"Training Accuracy: {model_nb.score(X_train, y_train)}")
        print(f"Validation Accuracy: {model_nb.score(X_val, y_val)}")
        
        print(f"Saving model to {model_output_directory}")
        joblib.dump(model_nb, model_output_directory)

# Function to evaluate the Naive Bayes Classifier model
def evaluation_nb(model_path, test_features_data, test_labels_data, evaluation_output_path):
        """
        This function evaluates the model and saves the evaluation report to the evaluation_output_path.
        """
        model = joblib.load(model_path)
        
        X_test = pd.read_csv(test_features_data, header=None)
        y_test = pd.read_csv(test_labels_data, header=None)
        
        y_pred = model.predict(X_test)
        
        print("Test Set Metrics")
        print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
        print(f"Precision: {precision_score(y_test, y_pred)}")
        print(f"Recall: {recall_score(y_test, y_pred)}")
        print(f"F1 Score: {f1_score(y_test, y_pred)}")
        print(f"ROC AUC: {roc_auc_score(y_test, y_pred)}")
        print(f"Confusion Matrix: \n{confusion_matrix(y_test, y_pred)}")
        print(f"Classification Report: \n{classification_report(y_test, y_pred)}")
        
        report_dict = {
            "classification_metrics": {
                "accuracy": {
                    "value": accuracy_score(y_test, y_pred)
                },
                "precision": {
                    "value": precision_score(y_test, y_pred)
                },
                "recall": {
                    "value": recall_score(y_test, y_pred)
                },
                "f1": {
                    "value": f1_score(y_test, y_pred)
                },
                "roc_auc": {
                    "value": roc_auc_score(y_test, y_pred)
                }
            }
        }
        
        with open(evaluation_output_path, "w") as f:
            f.write(json.dumps(report_dict))
