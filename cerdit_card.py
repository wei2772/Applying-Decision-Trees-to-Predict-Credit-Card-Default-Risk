# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 04:42:41 2024

@author: Kuanwei
"""

import pandas as pd
import random
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
from imblearn.ensemble import EasyEnsembleClassifier
from imblearn.pipeline import Pipeline
import numpy as np

# Load the data
credit_data = pd.read_csv("default of credit card clients.csv")

# Data preprocessing
credit_data = credit_data.drop(columns=['ID'])
credit_data['default payment next month'] = credit_data['default payment next month'].map({1: "actu.yes", 0: "actu.no"})

random.seed(123)
m = credit_data.shape[1]
n = credit_data.shape[0]
cv = 3
num = int(n/cv)
tr = 15

column_names=["Accuracy", "Precision", "Recall", "F1_score"]
row_names=["1st", "2nd", "3rd", "Ave"]
table_cart_tr = np.zeros((cv+1, 4))
table_cart_tr = pd.DataFrame(table_cart_tr, columns=column_names, index=row_names)

table_cart_ts = np.zeros((cv+1, 4))
table_cart_ts = pd.DataFrame(table_cart_ts, columns=column_names, index=row_names)

table_cart_SE_tr = np.zeros((cv+1, 4))
table_cart_SE_tr = pd.DataFrame(table_cart_SE_tr, columns=column_names, index=row_names)

table_cart_SE_ts = np.zeros((cv+1, 4))
table_cart_SE_ts = pd.DataFrame(table_cart_SE_ts, columns=column_names, index=row_names)

table_C50_tr = np.zeros((cv+1, 4))
table_C50_tr = pd.DataFrame(table_C50_tr, columns=["Accuracy", "Precision", "Recall", "F_measure"], index=list(range(1, cv+2)))

table_C50_ts = np.zeros((cv+1, 4))
table_C50_ts = pd.DataFrame(table_C50_ts, columns=["Accuracy", "Precision", "Recall", "F_measure"], index=list(range(1, cv+2)))

table_C50_SE_tr = np.zeros((cv+1, 4))
table_C50_SE_tr = pd.DataFrame(table_C50_SE_tr, columns=["Accuracy", "Precision", "Recall", "F_measure"], index=list(range(1, cv+2)))

table_C50_SE_ts = np.zeros((cv+1, 4))
table_C50_SE_ts = pd.DataFrame(table_C50_SE_ts, columns=["Accuracy", "Precision", "Recall", "F_measure"], index=list(range(1, cv+2)))


table_cart_EE_tr = np.zeros((cv+1, 4))
table_cart_EE_tr = pd.DataFrame(table_cart_EE_tr, columns=column_names, index=row_names)

table_cart_EE_ts = np.zeros((cv+1, 4))
table_cart_EE_ts = pd.DataFrame(table_cart_EE_ts, columns=column_names, index=row_names)

for i in range(cv):
    j = i * num
    k = (i+1) * num
    ts_credit = credit_data.iloc[j:k]
    tr_credit = credit_data.drop(credit_data.index[j:k])


    # CART
    credit_cart = DecisionTreeClassifier(min_samples_split=10, min_samples_leaf=50, max_depth=4)
    credit_cart.fit(tr_credit.drop(columns=['default payment next month']), tr_credit['default payment next month'])

    tr_cart = credit_cart.predict(tr_credit.drop(columns=['default payment next month']))
    confusion_cart_tr = pd.crosstab(tr_credit['default payment next month'], tr_cart)
    table_cart_tr.iloc[i] = [accuracy_score(tr_credit['default payment next month'], tr_cart),
                             precision_score(tr_credit['default payment next month'], tr_cart, pos_label='actu.yes'),
                             recall_score(tr_credit['default payment next month'], tr_cart, pos_label='actu.yes'),
                             f1_score(tr_credit['default payment next month'], tr_cart, pos_label='actu.yes')]

    ts_cart = credit_cart.predict(ts_credit.drop(columns=['default payment next month']))
    confusion_cart_ts = pd.crosstab(ts_credit['default payment next month'], ts_cart)
    table_cart_ts.iloc[i] = [accuracy_score(ts_credit['default payment next month'], ts_cart),
                             precision_score(ts_credit['default payment next month'], ts_cart, pos_label='actu.yes'),
                             recall_score(ts_credit['default payment next month'], ts_cart, pos_label='actu.yes'),
                             f1_score(ts_credit['default payment next month'], ts_cart, pos_label='actu.yes')]

    # SMOTE-ENN CART
    under_sampler = RandomUnderSampler(sampling_strategy={"actu.no": 5000})
    smote_enn = SMOTEENN(sampling_strategy={"actu.yes": 5000})
    pipeline = Pipeline([('under_sampler', under_sampler), ('smote_enn', smote_enn)])

    tr_credit_X_resampled, tr_credit_y_resampled = pipeline.fit_resample(tr_credit.drop(columns=['default payment next month']), tr_credit['default payment next month'])

    credit_cart = DecisionTreeClassifier(min_samples_split=10, min_samples_leaf=50, max_depth=4)
    credit_cart.fit(tr_credit_X_resampled, tr_credit_y_resampled)

    tr_cart_SE = credit_cart.predict(tr_credit.drop(columns=['default payment next month']))
    confusion_cart_SE_tr = pd.crosstab(tr_credit['default payment next month'], tr_cart_SE)
    table_cart_SE_tr.iloc[i] = [accuracy_score(tr_credit['default payment next month'], tr_cart_SE),
                                precision_score(tr_credit['default payment next month'], tr_cart_SE, pos_label='actu.yes'),
                                recall_score(tr_credit['default payment next month'], tr_cart_SE, pos_label='actu.yes'),
                                f1_score(tr_credit['default payment next month'], tr_cart_SE, pos_label='actu.yes')]

    ts_cart_SE = credit_cart.predict(ts_credit.drop(columns=['default payment next month']))
    confusion_cart_SE_ts = pd.crosstab(ts_credit['default payment next month'], ts_cart_SE)
    table_cart_SE_ts.iloc[i] = [accuracy_score(ts_credit['default payment next month'], ts_cart_SE),
                                precision_score(ts_credit['default payment next month'], ts_cart_SE, pos_label='actu.yes'),
                                recall_score(ts_credit['default payment next month'], ts_cart_SE, pos_label='actu.yes'),
                                f1_score(ts_credit['default payment next month'], ts_cart_SE, pos_label='actu.yes')]


    # EasyEnsemble
    easy_ensemble = EasyEnsembleClassifier(n_estimators=tr)
    easy_ensemble.fit(tr_credit.drop(columns=['default payment next month']), tr_credit['default payment next month'])

    tr_pred = easy_ensemble.predict(tr_credit.drop(columns=['default payment next month']))
    ts_pred = easy_ensemble.predict(ts_credit.drop(columns=['default payment next month']))

    table_cart_EE_tr.iloc[i] = [accuracy_score(tr_credit['default payment next month'], tr_pred),
                                 precision_score(tr_credit['default payment next month'], tr_pred, pos_label='actu.yes'),
                                 recall_score(tr_credit['default payment next month'], tr_pred, pos_label='actu.yes'),
                                 f1_score(tr_credit['default payment next month'], tr_pred, pos_label='actu.yes')]


    table_cart_EE_ts.iloc[i] = [accuracy_score(ts_credit['default payment next month'], ts_pred),
                                 precision_score(ts_credit['default payment next month'], ts_pred, pos_label='actu.yes'),
                                 recall_score(ts_credit['default payment next month'], ts_pred, pos_label='actu.yes'),
                                 f1_score(ts_credit['default payment next month'], ts_pred, pos_label='actu.yes')]

# Calculate mean values
for j in range(4):
    table_cart_tr.iloc[cv, j] = table_cart_tr.iloc[:cv, j].mean()
    table_cart_ts.iloc[cv, j] = table_cart_ts.iloc[:cv, j].mean()
    table_cart_SE_tr.iloc[cv, j] = table_cart_SE_tr.iloc[:cv, j].mean()
    table_cart_SE_ts.iloc[cv, j] = table_cart_SE_ts.iloc[:cv, j].mean()
    table_cart_EE_tr.iloc[cv, j] = table_cart_EE_tr.iloc[:cv, j].mean()
    table_cart_EE_ts.iloc[cv, j] = table_cart_EE_ts.iloc[:cv, j].mean()

print("CART Training Table:")
print(table_cart_tr)
print("\nCART Testing Table:")
print(table_cart_ts)
print("\nCART+SMOTEENN+RUS Training Table:")
print(table_cart_SE_tr)
print("\nCART+SMOTEENN+RUS Testing Table:")
print(table_cart_SE_ts)
print("\nEasy Ensemble Training Table:")
print(table_cart_EE_tr)
print("\nEasy Ensemble Testing Table:")
print(table_cart_EE_ts)
