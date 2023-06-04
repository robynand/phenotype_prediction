import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
from skopt import BayesSearchCV
import skopt
from sklearn.metrics import make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['figure.facecolor']='white'
from sklearn.model_selection import KFold
from math import sqrt
from skopt.space import Real, Categorical, Integer
import xgboost as xgb
from skopt.searchcv import BayesSearchCV
import pickle
import sklearn

#import datasets
SNPs = pd.read_csv("SNPs.csv", sep= "\t", index_col=0)
phenotypes = pd.read_csv('phenotypes.csv', sep = "\t", index_col=0)
genandphen= pd.merge(SNPs4, phenotypes, left_index=True, right_index=True, how='inner')
data_phm = genandphen.iloc[:, 0:16624]
target_phm = genandphen.iloc[:, 16624:]

#split train and test datasets
X_train, X_holdout, y_train, y_holdout = train_test_split(data_enc, target, test_size=0.2, random_state=42, shuffle=True)
X_train = X_train.to_numpy()
y_train = y_train.to_numpy()

#set up optimisation
space ={'learning_rate': Real(0.01, 1.0, 'log-uniform'),
        'min_child_weight': Integer(0, 10),
        'max_depth': Integer(0, 50),
        'max_delta_step': Integer(0, 20),
        'subsample': Real(0.01, 1.0, 'uniform'),
        'colsample_bytree': Real(0.01, 1.0, 'uniform'),
        'colsample_bylevel': Real(0.01, 1.0, 'uniform'),
        'reg_lambda': Real(1e-9, 1000, 'log-uniform'),
        'reg_alpha': Real(1e-9, 1.0, 'log-uniform'),
        'gamma': Real(1e-9, 0.5, 'log-uniform'),
        'min_child_weight': Integer(0, 5),
        'n_estimators': Integer(50, 200),
        'scale_pos_weight': Real(1e-6, 500, 'log-uniform')}

def on_step(optim_result):
    """
    Callback meant to view scores after
    each iteration while performing Bayesian
    Optimization in Skopt"""
    score = xgb_bayes_search.best_score_
    print("best score: %s" % score)
    if score >= 0.98:
        print('Interrupting!')
        return True

xgbreg = xgb.XGBRegressor()
xgb_bayes_search = BayesSearchCV(xgbreg, space, n_iter=9, # specify how many iterations
                                    scoring=None, n_jobs=-1, cv=5, verbose=3, random_state=42, n_points=12)
#import datetime
#print(datetime.datetime.now().time())
xgb_bayes_search.fit(X_train, y_train, callback=on_step)
#print(datetime.datetime.now().time())
# print(xgb_bayes_search.best_params_)

best_params = xgb_bayes_search.best_params_
xgbreg = xgb.XGBRegressor(**best_params)
print(xgbreg)

def eval_k_fold(m, x, y, k):
    #model: xgboost model, should be with the best params available
    #x: input data (eg. all samples and SNPS)
    #y: labels
    #k: number of folds for cross validation
    cv = KFold(n_splits=k,shuffle=True)
  #  fig1 = plt.figure(figsize=[12,12])

   # tprs = []
   # aucs = []
    results = []
   # mean_fpr = np.linspace(0,1,100)
    low = 100
    best = m
    i = 1
    for train,test in cv.split(x,y):
        #print(y[test])
        m.fit(x[train],y[train].ravel())
        print("fitting done. Processing fold accuracy + checking best model")
        #predictions = [round(value) for value in y_pred]
        #sees how accurate the model was when testing the test set
        all_preds = [x for x in m.predict(x[test])]
        ss = sqrt(mean_squared_error(all_preds, y[test]))
        rr = r2_score(all_preds, y[test])
        mm = np.mean(y[test])
        error_mean = ((ss/mm)*100)
        print("R^2 Value is: " + str(rr))
        print("RMSE for dataset is:" +str(ss) + "& mean of this fold is " + str(mm))
        print("this is "+ str((ss/mm)*100) + "% of the mean pheno data")
        if(error_mean < low):
            low = error_mean
            best = m
        results.append(error_mean)
        i= i+1
    print("Training Testing Accuracy: %.2f%% (%.2f%%)" % (np.mean(results), np.std(results)))
    return best


#Training final model
best_model = eval_k_fold(xgbreg, X_train, y_train, 10)
pickle.dump(best_model, open("Marcroft_1perc_10fold_XGB222.pickle.dat", "wb"))

#Evaluation 
X_holdout = X_holdout.to_numpy()
y_holdout = y_holdout.to_numpy()

all_preds = [x for x in best_model.predict(X_holdout)]
ss = sqrt(mean_squared_error(all_preds, y_holdout))
rr = r2_score(all_preds, y_holdout)
mm = np.mean(y_holdout)
error_mean = ((ss/mm)*100)
print("R^2 Value of Holdout: %.2f" % rr)
print("RMSE of Holdout: %.2f" % ss)
print("Mean of Holdout: %.2f" % mm)
print("this is "+ str((ss/mm)*100) + "% of the mean pheno data")

plot_x, plot_y = list(), list()

y_holdout = y_holdout.ravel()

for counter, i in enumerate(y_holdout):
    if counter <= 5:
        print(counter, i, all_preds[counter])
    #zoom in a bit closer
    if(all_preds[counter] > 1):
        plot_x.append(i)
        plot_y.append(all_preds[counter])
#figure
fig1 = sns.regplot(x="actual", y="predicted", data=thisplot).set_title('XGBoost predicted vs actual 10% window')

#plot important features
y_holdout = y_holdout.tolist()
out_df = pd.DataFrame(data={"y_orig": y_holdout, "y_pred": all_preds})
out_df.to_csv("y_orig_vs_y_pred_trait13.csv", sep=',',index=False)

from xgboost import plot_importance
from matplotlib import pyplot
#best_model = pickle.load(open("Oil_kfold_10_tt_from_all.pickle.dat", "rb"))
plt.figure(figsize = (20, 20))
plot_importance(best_model, max_num_features=15, importance_type='gain', height=0.3)
pyplot.show()

f_names = ['f' + str(i) for i in range(len(data_enc.columns))]
my_dict = best_model.get_booster().get_score(importance_type="gain")

new_dict = {}
for key in my_dict:
    ind = f_names.index(key)
    new_dict[data_enc.columns[ind]] = my_dict[key]

new_fi = pd.Series(new_dict)
df = new_fi.to_frame()
df = df.rename(columns = {0:'F_Score(GAIN)'})

#plot dataframe now with top 20 SNPS
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
figure(num=None, figsize=(8, 8), dpi=100, facecolor='w', edgecolor='k')
indexes = df.nlargest(20, "F_Score(GAIN)").index
values = df.nlargest(20, "F_Score(GAIN)").values.ravel()
indexes = indexes[::-1]
values = values[::-1]
plt.barh(indexes, values)
plt.title('Title)')
plt.ylabel('SNP Label')
plt.xlabel('Relative F_Score (GAIN)')
plt.savefig('file.png')
       
