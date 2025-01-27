'''
 This python file is for reading the data, training the model and saving the model.
 Steps : 
 1- Reading the data
 2- Training the model
 3- Saving the model
''' 

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score , confusion_matrix, classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
import joblib

df = pd.read_csv('diabetes.csv')
X= df.drop('Outcome', axis=1)
Y= df['Outcome']

smote = SMOTE(random_state=42)
X, Y = smote.fit_resample(X, Y)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=42)

'''
Pripeline : 
1- imputer : to fill missing values
2- scaler : to normalize the data
3- classifier : to train the model
'''

pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier())
])

pipeline.fit(X_train, Y_train)
Y_pred = pipeline.predict(X_test)

'''
Pripeline : 
1- imputer : to fill missing values
2- scaler : to normalize the data
3- classifier : to train the model
'''

pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier())
])

pipeline.fit(X_train, Y_train)
Y_pred = pipeline.predict(X_test)

print(f"Pipeline Accuracy ,{accuracy_score(Y_test, Y_pred)}")

# Craeting Parameter Grid
# Hyperparameter Tuning
param_grid = {
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__n_estimators': [20, 50, 100],
    'classifier__max_depth': [None, 10, 20 ]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5)
grid_search.fit(X_train, Y_train)

best_params = grid_search.best_params_
print("Best Parameters:", best_params)      
best_model = grid_search.best_estimator_
Y_pred = best_model.predict(X_test)
print(f"Model Accuracy , {accuracy_score(Y_test, Y_pred)}")

# save the pipeline
joblib.dump(best_model, 'model_pipeline.pkl')