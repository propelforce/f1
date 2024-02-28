# %%
import os
import pandas as pd
import numpy as np
import lightgbm as lgb
import pickle
from sklearn.model_selection import TimeSeriesSplit,GridSearchCV
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, classification_report

pd.set_option('display.max_columns', None)


# %%

main_directory = os.getcwd()

features = r'data\interim\base_dataset_features.csv'
csv_files_path = os.path.join(main_directory, features)

df_features = pd.read_csv(features)
# Target variable
df_features['Top_3_finish'] = df_features['final_position'].le(3).astype(int)

# %%
column_list = list(df_features.columns)
drop_columns = ['race_date', 'qualifyId', 'wins', 'driver_race_points', 'constructor_race_points', 'driver_season_position', 'constructor_season_points', 'final_position', 'final_status']

def difference(a, b):
  """Return the difference of two lists."""
  result = []
  for item in a:
    if item not in b:
      result.append(item)
  return result

final_columns = difference(column_list, drop_columns)

# %%
X = df_features[df_features['year'] >= 2013].reset_index(drop=True)
X = X[final_columns]
y = X.Top_3_finish
X = X.drop(['Top_3_finish'], axis=1)

# %%
# #GridSearch

# # Initialize a LightGBM classifier
# clf = lgb.LGBMClassifier()

# # Parameters to search
# param_grid = {
#     'num_leaves': [20, 31, 40],
#     'learning_rate': [0.05, 0.1, 0.2],
#     'n_estimators': [50, 100, 200],
#     'objective': ['binary'],  # Specify binary classification objective
#     'metric': ['binary_logloss'],  # Specify binary logloss as the metric
#     'boosting_type': ['gbdt']  # Specify boosting type as gbdt
# }

# # Initialize a time series cross-validator
# tscv = TimeSeriesSplit(n_splits=5)

# # Initialize GridSearchCV
# # Use 'precision_weighted' and 'recall_weighted' as scoring metrics
# grid_search = GridSearchCV(clf, param_grid, scoring=['precision_weighted', 'recall_weighted'], refit='precision_weighted', cv=tscv, verbose=2)

# # Perform GridSearchCV
# grid_search.fit(X, y)

# # Get the best parameters
# best_params = grid_search.best_params_

# print(f"Best Parameters: {best_params}")

# # Train the classifier with the best parameters
# best_clf = lgb.LGBMClassifier(**best_params)
# best_clf.fit(X, y)

# # Make predictions
# y_pred = best_clf.predict(X)

# # Compute metrics on the full dataset
# conf_mat = confusion_matrix(y, y_pred)
# precision = precision_score(y, y_pred, average='micro')
# recall = recall_score(y, y_pred, average='micro')
# f1 = f1_score(y, y_pred, average='weighted')

# print(f"Confusion Matrix:\n{conf_mat}")
# print(f"Precision: {precision}")
# print(f"Recall: {recall}")
# print(f"F1 Score: {f1}")


# %%
#Train-test split
train_data = df_features[(df_features['year'] >= 2013) & (df_features['race_date'] < '2023-04-02')].reset_index(drop=True)
test_data = df_features[(df_features['year'] == 2023) & (df_features['race_date'] == '2023-04-02')].reset_index(drop=True)

# train_data = df_features[(df_features['year'] >= 2016) & (df_features['year'] < 2023)].reset_index(drop=True)
# test_data = df_features[(df_features['year'] == 2023)].reset_index(drop=True)

train_data = train_data[final_columns]
test_data = test_data[final_columns]

# Features and target variable
X_train = train_data.drop(['Top_3_finish'], axis=1)
y_train = train_data['Top_3_finish']

X_test = test_data.drop(['Top_3_finish'], axis=1)
y_test = test_data['Top_3_finish']

# %%
# Define LightGBM parameters
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 20,
    'learning_rate': 0.1,
    'n_estimators' : 100
}

# Create LightGBM datasets
train_dataset = lgb.Dataset(X_train, label=y_train)
test_dataset = lgb.Dataset(X_test, label=y_test, reference=train_dataset)

print('Model training started')

# Train the LightGBM model
num_round = 100
bst = lgb.train(params, train_dataset, num_round, valid_sets=[test_dataset])

print ('Model training completed')

# Get feature importance
feature_importance = bst.feature_importance(importance_type='gain')
feature_importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importance})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Display feature importance
print("Feature Importance scores:")
print(feature_importance_df)

# Make predictions on the test set
y_pred_proba = bst.predict(X_test, num_iteration=bst.best_iteration)
y_pred_binary = (y_pred_proba > 0.5).astype(int)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred_binary))
print("Precision:", precision_score(y_test, y_pred_binary))
print("Recall:", recall_score(y_test, y_pred_binary))
print("\nClassification Report:\n", classification_report(y_test, y_pred_binary))

# %%
test_data['pred_result'] = y_pred_binary
test_data['Top_3_finish'] = y_test

train_data['Top_3_finish'] = y_train

final_path = r'data\final'

file_name = r'train_data.csv'
csv_file_path = os.path.join(main_directory, final_path,file_name)
train_data.to_csv(csv_file_path,index=False)


file_name = r'test_data.csv'
csv_file_path = os.path.join(main_directory, final_path,file_name)
test_data[['driverId','circuitId','raceId','year','race_month','race_day','Top_3_finish','pred_result']].to_csv(csv_file_path,index=False)

file_name = r'feature_importance_df.csv'
csv_file_path = os.path.join(main_directory, final_path,file_name)
feature_importance_df.to_csv(csv_file_path,index=False)

print('Written predictions and feature importance scores to the output path')

# %%
final_path = r'models'
file_name = r'f1-race-top3-predictor.pkl'
model_file_path = os.path.join(main_directory, final_path,file_name)

with open(model_file_path, 'wb') as file:
    pickle.dump(bst, file)
    
print('Model Pickled successfully')


