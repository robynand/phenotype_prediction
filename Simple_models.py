#import packages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
from skopt import BayesSearchCV
import skopt
from sklearn.metrics import make_scorer
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

#import datasets
SNPs = pd.read_csv("SNPs.csv", sep= "\t", index_col=0)
phenotypes = pd.read_csv('phenotypes.csv', sep = "\t", index_col=0)
genandphen= pd.merge(SNPs4, phenotypes, left_index=True, right_index=True, how='inner')
data_phm = genandphen.iloc[:, 0:16624]
target_phm = genandphen.iloc[:, 16624:]

#spit train and test datasets
X_train_phm, X_test_phm, y_train_phm, y_test_phm = train_test_split(data_phm, target_phm, random_state=42) 

#create dataframe with y_test values
y_test_phm_indexed = y_test_phm.reset_index()
y_pred_phm_df = pd.DataFrame(y_pred_phm)

#Ridge Regression
#import and train
from sklearn.linear_model import RidgeCV
clf = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1, 2, 5, 10], cv=5).fit(X_train_phm, y_train_phm)
clf.score(X_train_phm, y_train_phm)
y_pred_ridge_phm = clf.predict(X_test_phm)
#evaluation
r2_score(y_test_phm, y_pred_phm)
mean_squared_error(y_test_phm, y_pred_phm, squared=False)
#figure
y_pred_true_phm = pd.merge(y_test_phm_indexed, y_pred_phm_df, left_index=True, right_index=True, how='outer')
fig1 = sns.regplot(data=y_pred_true_phm2, x="phoma_at_harvest", y='predicted')
fig1.set_title('Ridge Regression', fontsize=15)
fig1.set_xlabel('Actual')
fig1.set_ylabel('Predicted')


#Lasso Regression
#import and train
from sklearn.linear_model import LassoCV
clf_lasso =  LassoCV(cv=5, random_state=0, n_alphas=500).fit(X_train_phm, y_train_phm)
y_pred_lasso_phm = clf_lasso.predict(X_test_phm)
#evaluation
r2_score(y_test_phm, y_pred_lasso_phm)
mean_squared_error(y_test_phm, y_pred_lasso_phm, squared=False)
#figure
y_pred_true_phm = pd.merge(y_test_phm_indexed, y_pred_phm_df, left_index=True, right_index=True, how='outer')
y_pred_true_phm2 = y_pred_true_phm.rename({0: 'predicted'}, axis='columns')
fig1 = sns.regplot(data=y_pred_true_phm2, x="phoma_at_harvest", y='predicted')
fig1.set_title('Lasso Regression', fontsize=15)
fig1.set_xlabel('Actual')
fig1.set_ylabel('Predicted')
#determine most important  features
coefficients = clf_lasso.coef_

importance = np.abs(coefficients)

importance2  = pd.DataFrame(importance)
columns_list = data_enc.columns.to_list()
columns_df = pd.DataFrame(columns_list)
col_and_imp = pd.merge(columns_df, importance2, left_index=True, right_index=True, how='inner')
col_and_imp.rename(columns={ '0_x' : 'Variable', '0_y': 'Coefficient'}, inplace=True)
col_and_imp.sort_values(by="Coefficient", ascending=False, inplace=True)
col_and_imp2 = col_and_imp.set_index('Variable')

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
figure(num=None, figsize=(8, 8), dpi=300, facecolor='w', edgecolor='k')
indexes = col_and_imp2.nlargest(20, "Coefficient").index
values = col_and_imp2.nlargest(20, "Coefficient").values.ravel()
indexes = indexes[::-1]
values = values[::-1]
plt.barh(indexes, values)
plt.title('Lasso Regression SNP Importance', fontsize=15)
plt.ylabel('SNP')
plt.xlabel('Coefficient')
plt.tight_layout()
plt.savefig('file.png')

#Support Vector Machine
#import train
from sklearn.svm import LinearSVR
model_to_set = LinearSVR(max_iter=5000)
parameters = { 'epsilon': [0.5, 1, 1.5, 2, 2.5, 3],                                                                              }
model_tunning = GridSearchCV(model_to_set, param_grid=parameters, scoring='r2', cv=5)
model_tunning.fit(X_train_phm, y_train_phm)
print(model_tunning.best_score_)
print(model_tunning.best_params_)
SVM_linear =  LinearSVR(epsilon=2)
SVM_linear.fit(X_train_phm, y_train_phm)
y_pred_SVM_phm = SVM_linear.predict(X_test_phm)
#evaluation
r2_score(y_test_phm, y_pred_SVM_phm)
mean_squared_error(y_test_phm, y_pred_SVM_phm, squared=False)
#figure
y_pred_true_phm = pd.merge(y_test_phm_indexed, y_pred_phm_df, left_index=True, right_index=True, how='outer')
y_pred_true_phm2 = y_pred_true_phm.rename({0: 'predicted'}, axis='columns')
fig1 = sns.regplot(data=y_pred_true_phm2, x="phoma_at_harvest", y='predicted')
fig1.set_title('Support Vector Regressor', fontsize=15)
fig1.set_xlabel('Actual')
fig1.set_ylabel('Predicted')

#Random Forest Regressor
#import train
from sklearn.ensemble import RandomForestRegressor
model_to_set = RandomForestRegressor(n_jobs=-1 )
parameters = {
        'max_depth': [3, 5, 7, 9, 11],
        'n_estimators': [50, 150, 200, 250, 300, 350],                                                                   
        }
model_tunning = GridSearchCV(model_to_set, param_grid=parameters,
                             scoring='neg_mean_squared_error', cv=5)
model_tunning.fit(X_train_phm, y_train_phm)
print(model_tunning.best_score_)
print(model_tunning.best_params_)
rand = RandomForestRegressor(n_jobs=-1, max_depth=7, n_estimators = 250)
rand.fit(X_train_phm, y_train_phm)
y_pred_rand_phm = rand.predict(X_test_phm)
#evaluation
r2_score(y_test_phm, y_pred_rand_phm)
mean_squared_error(y_test_phm, y_pred_rand_phm, squared=False)
#figure
y_pred_true_phm = pd.merge(y_test_phm_indexed, y_pred_phm_df, left_index=True, right_index=True, how='outer')
y_pred_true_phm2 = y_pred_true_phm.rename({0: 'predicted'}, axis='columns')
fig1 = sns.regplot(data=y_pred_true_phm2, x="phoma_at_harvest", y='predicted')
fig1.set_title('Random Forest', fontsize=15)
fig1.set_xlabel('Actual')
fig1.set_ylabel('Predicted')
#determine best features
rand_importance = pd.DataFrame({'Variable':data_enc.columns,
              'Importance':rand.feature_importances_}).sort_values('Importance', ascending=False)
rand_importance2 = rand_importance.set_index('Variable')
rand_importance2.head()
rand_importance2 = rand_imp.set_index('Variable')
rand_importance2.head()
#figure
figure(num=None, figsize=(8, 8), dpi=300, facecolor='w', edgecolor='k')
indexes = rand_importance2.nlargest(20, "Importance").index
values = rand_importance2.nlargest(20, "Importance").values.ravel()
indexes = indexes[::-1]
values = values[::-1]
plt.barh(indexes, values)
plt.title('Random Forest SNP Importance')
plt.ylabel('SNP')
plt.xlabel('Feature Importance')
plt.tight_layout()
plt.savefig('file.png')


#Decision Tree Regressor
#import and train
from sklearn.tree import DecisionTreeRegressor
model_to_set = DecisionTreeRegressor()
parameters = {
        'max_depth': [3, 5, 7, 9, 11],                                                              
        }
model_tunning = GridSearchCV(model_to_set, param_grid=parameters,
                             scoring='neg_mean_squared_error', cv=5)
model_tunning.fit(X_train_phm, y_train_phm)
print(model_tunning.best_score_)
print(model_tunning.best_params_)
dec = DecisionTreeRegressor(max_depth=3)
dec.fit(X_train_phm, y_train_phm)
y_pred_dec_phm = dec.predict(X_test_phm)
#evaluation
r2_score(y_test_phm, y_pred_dec_phm)
mean_squared_error(y_test_phm, y_pred_dec_phm, squared=False)
#figure
y_pred_true_phm = pd.merge(y_test_phm_indexed, y_pred_phm_df, left_index=True, right_index=True, how='outer')
y_pred_true_phm2 = y_pred_true_phm.rename({0: 'predicted'}, axis='columns')
fig1 = sns.regplot(data=y_pred_true_phm2, x="phoma_at_harvest", y='predicted')
fig1.set_title('Decision Tree', fontsize=15)
fig1.set_xlabel('Actual')
fig1.set_ylabel('Predicted')

