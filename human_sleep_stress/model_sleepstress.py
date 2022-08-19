### EDA

## import library
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

import os

print(os.getcwd())
path = "/Users/mac/Desktop/kaggle4th_flask_ml/kaggle4th_flask/human_stress_sleep/"



## load data
data = pd.read_csv(path + "data/SaYoPillow.csv")
print(data.head())
print(data.shape)

#       sr      rr       t      lm  ...    rem   sr.1     hr  sl
# 0  93.80  25.680  91.840  16.600  ...  99.60  1.840  74.20   3
# 1  91.64  25.104  91.552  15.880  ...  98.88  1.552  72.76   3
# 2  60.00  20.000  96.000  10.000  ...  85.00  7.000  60.00   1
# 3  85.76  23.536  90.768  13.920  ...  96.92  0.768  68.84   3
# 4  48.12  17.248  97.872   6.496  ...  72.48  8.248  53.12   0

# (630, 9)


data.columns=['snoring_rate', 'respiration_rate', 'body_temperature', 'limb_movement', 'blood_oxygen', \
             'eye_movement', 'sleeping_hours', 'heart_rate', 'stress_level']
print(data.head())
print(data.info())

#    snoring_rate  respiration_rate  ...  heart_rate  stress_level
# 0         93.80            25.680  ...       74.20             3
# 1         91.64            25.104  ...       72.76             3
# 2         60.00            20.000  ...       60.00             1
# 3         85.76            23.536  ...       68.84             3
# 4         48.12            17.248  ...       53.12             0

# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 630 entries, 0 to 629
# Data columns (total 9 columns):
#  #   Column            Non-Null Count  Dtype  
# ---  ------            --------------  -----  
#  0   snoring_rate      630 non-null    float64
#  1   respiration_rate  630 non-null    float64
#  2   body_temperature  630 non-null    float64
#  3   limb_movement     630 non-null    float64
#  4   blood_oxygen      630 non-null    float64
#  5   eye_movement      630 non-null    float64
#  6   sleeping_hours    630 non-null    float64
#  7   heart_rate        630 non-null    float64
#  8   stress_level      630 non-null    int64  
# dtypes: float64(8), int64(1)



## 8 features related to sleep and 1 target



## Distribution of data

# 'stress_level' value counts
print(data['stress_level'].value_counts())

# 3    126
# 1    126
# 0    126
# 2    126
# 4    126
# Name: stress_level, dtype: int64


# variable correlations of data
plt.figure(figsize=(15,10))
sns.heatmap(data.corr(), annot=True)
plt.title('Heatmap of Variable Correlations')
plt.show()

# see the target variable 'stress_level' correlations with the other variables
# correlation of 'snoring_rate' and 'stress_level'
plt.figure(figsize=(15,5))
sns.lineplot(x='snoring_rate', y='stress_level', data=data)
plt.title('Snoring rate vs Stress level')
plt.xlabel('Snoring rate')
plt.ylabel('Stress level')
plt.show()

# correlation of 'respiration_rate' and 'stress_level'
plt.figure(figsize=(15,5))
sns.lineplot(x='respiration_rate', y='stress_level', data=data)
plt.title('Respiration rate vs Stress level')
plt.xlabel('Respiration rate')
plt.ylabel('Stress level')
plt.show()

# correlation of 'snoring_rate' and 'respiration_rate'
plt.figure(figsize=(10,5))
sns.scatterplot(x='snoring_rate', y='respiration_rate', data=data, hue='stress_level', palette='deep')
plt.title('Snoring rate vs Respiration rate')
plt.xlabel('Snoring rate')
plt.ylabel('Respiration rate')
plt.show()

# correlation of 'body_temperature' and 'stress_level'
plt.figure(figsize=(15,5))
sns.lineplot(x='body_temperature', y='stress_level', data=data)
plt.title('Body temperature vs Stress level')
plt.xlabel('Body temperature')
plt.ylabel('Stress level')
plt.show()

# correlation of 'limb_movement' and 'stress_level'
plt.figure(figsize=(15,5))
sns.lineplot(x='limb_movement', y='stress_level', data=data)
plt.title('Limb movement vs Stress level')
plt.xlabel('Limb movement')
plt.ylabel('Stress level')
plt.show()

# correlation of 'body_temperature' and 'limb_movement'
plt.figure(figsize=(10,5))
sns.scatterplot(x='body_temperature', y='limb_movement', data=data, hue='stress_level', palette='deep')
plt.title('Body temperature vs Limb movement')
plt.xlabel('Body temperature')
plt.ylabel('Limb movement')
plt.show()

# correlation of 'blood_oxygen' and 'stress_level'
plt.figure(figsize=(15,5))
sns.lineplot(x='blood_oxygen', y='stress_level', data=data)
plt.title('Blood oxygen vs Stress level')
plt.xlabel('Blood oxygen')
plt.ylabel('Stress level')
plt.show()

# correlation of 'eye_movement' and 'stress level'
plt.figure(figsize=(15,5))
sns.lineplot(x='eye_movement', y='stress_level', data=data)
plt.title('Eye movement vs Stress level')
plt.xlabel('Eye movement')
plt.ylabel('Stress level')
plt.show()

# correlation of 'blood_oxygen' and 'eye_movement'
plt.figure(figsize=(10,5))
sns.scatterplot(x='blood_oxygen', y='eye_movement', data=data, hue='stress_level', palette='deep')
plt.title('Blood oxygen vs Eye movement')
plt.xlabel('Blood oxygen')
plt.ylabel('Eye movement')
plt.show()

# correlation of 'sleeping_hours' and 'stress_level'
plt.figure(figsize=(15,5))
sns.lineplot(x='sleeping_hours', y='stress_level', data=data)
plt.title('Sleeping hours vs Stress level')
plt.xlabel('Sleeping hours')
plt.ylabel('Stress level')
plt.show()

# correlation of 'heart rate' and 'stress_level'
plt.figure(figsize=(15,5))
sns.lineplot(x='heart_rate', y='stress_level', data=data)
plt.title('Heart rate vs Stress level')
plt.xlabel('Heart rate')
plt.ylabel('Stress level')
plt.show()

# correlation of 'sleeping_hours' and 'heart_rate'
plt.figure(figsize=(10,5))
sns.scatterplot(x='sleeping_hours', y='heart_rate', data=data, hue='stress_level', palette='deep')
plt.title('Sleeping hours vs Heart rate')
plt.xlabel('Sleeping hours')
plt.ylabel('Heart rate')
plt.show()





### Model building
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score


## select and split data
sel = ['snoring_rate', 'respiration_rate', 'body_temperature', 'limb_movement', 'blood_oxygen', 'eye_movement', 'sleeping_hours', 'heart_rate']
X = data[sel]
y = data['stress_level']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# (472, 8) (158, 8) (472,) (158,)


## baseline model
# Randomforest
model_rf = RandomForestClassifier(random_state=30).fit(X_train, y_train)
score = cross_val_score(model_rf, X_test, y_test, cv=5, scoring='accuracy')
print("cross_val_score - rf : ", np.mean(score))

# cross_val_score - rf :  0.9875

# Gradientboosting
model_gb = GradientBoostingClassifier(random_state=30).fit(X_train, y_train)
score = cross_val_score(model_gb, X_test, y_test, cv=5, scoring='accuracy')
print("cross_val_score - gb : ", np.mean(score))

# cross_val_score - gb :  0.9810483870967742


## improved model
Stress_Levels = ['Low', 'Medium Low', 'Medium', 'Medium High', 'High']
Feature_Importance = pd.DataFrame()
for i in range(0,5):
    data_Pred = data.copy()
    data_Pred['stress_level'] = data_Pred['stress_level'].apply(lambda x: 1 if x==i else 0)
    sel = ['snoring_rate', 'respiration_rate', 'body_temperature', 'limb_movement', 'blood_oxygen', 'eye_movement', 'sleeping_hours', 'heart_rate']
    X = data[sel]
    # X = data_Pred.drop('sl',axis=1)
    y = data['stress_level']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=100)
    
    Model = RandomForestClassifier(random_state=100, n_jobs=-1)
    
    params = {'n_estimators':[200],
              'min_samples_leaf':[2,5,10,20,30],
              'max_depth':[2,3,5,10,12,15,20],
              'max_features':[0.1,0.15,0.2,0.25,0.3,0.35,0.4]}
    
    grid_search = GridSearchCV(estimator=Model, param_grid=params, n_jobs=-1, verbose=1, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    # GridSearchCV의 refit으로 이미 학습이 된 estimator 반환
    Model_best = grid_search.best_estimator_
    
    # GridSearchCV의 best_estimator_는 이미 최적 하이퍼 파라미터로 학습이 됨
    y_train_pred = Model_best.predict(X_train)
    y_test_pred = Model_best.predict(X_test)
    
    print('Train Accuracy :', accuracy_score(y_train, y_train_pred))
    print('Test Accuracy :', accuracy_score(y_test, y_test_pred))
    
    Feature_Importance['Feature'] = X_train.columns
    Feature_Importance[Stress_Levels[i]] = Model_best.feature_importances_

# Train Accuracy : 1.0
# Test Accuracy : 0.9920634920634921


# feature importance
Feature_Importance.set_index('Feature',inplace=True)
Feature_Importance.head(10)

# view
plt.figure(figsize=(12,7))
sns.heatmap(Feature_Importance,annot=True)
plt.title('Heatmap of Feature Importances')
plt.show()

