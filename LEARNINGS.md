# What I Learned

## Exploring the data

Before building a model, I learned to inspect the dataset and understand what each column contains. Checking the number of rows and columns, data types, missing values, duplicates, and summary statistics helps catch issues early.

Visualizations also make it easier to see patterns in transaction amounts, account balances, transaction types, and the number of fraud cases.

## Preparing data for a model

Machine-learning models need a clear prediction target. In this project, `isFraud` is the target: `1` represents a fraudulent transaction and `0` represents a non-fraudulent transaction.

The transaction type is categorical, so it needs to be encoded before the model can use it. Numeric fields such as amount and balances need to be prepared as well. The app uses imputation for missing values, scaling for numeric features, and one-hot encoding for categories.

Transaction IDs such as `nameOrig` and `nameDest` are excluded because they identify individual accounts rather than describe a general transaction pattern. The existing `isFlaggedFraud` field is also excluded so the model does not rely on another fraud flag as an input.

## Training and evaluating a classifier

The app uses logistic regression as a starter classifier. It splits the data into training and test portions so the model can be evaluated on examples it did not train on.

Fraud datasets are usually imbalanced: there are many more legitimate transactions than fraudulent ones. A model could achieve high accuracy by predicting “not fraud” for nearly every row, while still missing most fraud. The app uses class weighting to give the less common fraud examples more influence during training.

Accuracy alone is not enough to judge a fraud model. Precision, recall, F1 score, confusion matrix, ROC-AUC, and average precision can give a fuller picture of false alarms and missed fraud cases.

## Building the prediction app

Streamlit makes it possible to build an interactive form with fields for transaction type, amount, time step, and sender and receiver balances. When I submit the form, the app arranges those values in the same feature columns used during training and asks the model for a prediction and estimated fraud probability.

The app trains the model from `AIML Dataset.csv` the first time a prediction is requested, then saves the trained model locally for later use. It uses up to 500,000 rows by default to keep the first training run more manageable.

## Limitations and next steps

A prediction is an estimate based on patterns in the training data. It is not proof that a transaction is fraudulent, and the demo should not be used to make real financial decisions.

For a stronger project, I would evaluate the trained model on a held-out test set, compare it with other classifiers, tune the decision threshold, and report precision and recall. I would also check that example inputs are realistic and consistent with the transaction type.

The CSV dataset is not included in this repository because of its size. To run the app, place `AIML Dataset.csv` in the project folder beside `fraud_detection.py`.
