# coding: utf-8
# ============================================================================
# Subject: Tópicos Especiais II (C318) 
#
# Theme: Diabetes Prediction
# 
# Students: Luana, Mariana, Sarah e Sinara
#
# Professor: Ricardo Augusto
# ----------------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import klib
from sklearn.model_selection import StratifiedKFold 

df = pd.read_csv('diabetes.csv')

df = df.replace(['Male','Female','Yes','No','Positive','Negative'],(1,0,1,0,1,0))

X = df.iloc[:, 0:16].to_numpy()
y = df.loc[:, 'class'].to_numpy()

skf = StratifiedKFold() #n_splitsint, default=5
skf.get_n_splits(X, y)

for train_index, test_index in skf.split(X,y):
    train_set = df.loc[train_index]
    test_index = df.loc[test_index]
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]


