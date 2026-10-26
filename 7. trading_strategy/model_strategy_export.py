from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

data_2017 = pd.read_csv('all_data_year_2017_a.csv')
data_2018 = pd.read_csv('all_data_year_2018_a.csv')
data_2019 = pd.read_csv('all_data_year_2019_a.csv')
data_2020 = pd.read_csv('all_data_year_2020_a.csv')
data_2021 = pd.read_csv('all_data_year_2021_a.csv')

def model_selection(X_train, X_test, y_train, y_test):
    # accuracy of different model
    names = [
    "Nearest Neighbors",
    "Linear SVM",
    "RBF SVM",
    # "Gaussian Process",
    "Decision Tree",
    "Random Forest",
    "Neural Net",
    "AdaBoost",
    "Naive Bayes",
    "QDA"]

    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025),
        SVC(gamma=2, C=1),
    #  GaussianProcessClassifier(1.0 * RBF(1.0)),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        MLPClassifier(alpha=1, max_iter=1000),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis()
    ]

    scores = []
    for name, clf in zip(names, classifiers):
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)

        scores.append(score)
        print(name,score)
    return scores

def function(a, b):
    if a>b:
        return 1
    else:
        return 0

# First roll: Use data of year 2017 to predict 2018.
# Prepare the data
y = data_2017.y_label
X = data_2017[['T-1_H-L/C', \
       'T-1_C-O/C', 'rolling_7_days_avg', 'rolling_14_days_avg', \
       'rolling_21_days_avg', 'Popularity_log','pred_label_0', \
       'pred_label_1', 'pred_label_2']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

model_selection(X_train, X_test, y_train, y_test)

# final model - Random Forest
RF = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
RF.fit(X_train, y_train)
RF.score(X_test, y_test)

# predict 2018 data
X_pred = data_2018[['T-1_H-L/C', \
       'T-1_C-O/C', 'rolling_7_days_avg', 'rolling_14_days_avg', \
       'rolling_21_days_avg', 'Popularity_log','pred_label_0', \
       'pred_label_1', 'pred_label_2']]

# which stocks to buy based on model strategy
y_pred = RF.predict(X_pred)
data_2018['ylabel_pred'] = y_pred

# which stocks to buy based on sentiment score strategy
data_2018['stmt_pred'] = data_2018.apply(lambda x: function(x.pred_label_0, x.pred_label_2), axis = 1)

# export the data of year 2018 using information of year 2017
data_2018.to_csv('data_2018_pred.csv')

#Second roll: Use data of 2017 and 2018 to predict 2019.
# combine data of year 2017 and 2018
data_1718 = pd.concat([data_2017,data_2018],ignore_index=True)

y = data_1718.y_label
X = data_1718[['T-1_H-L/C', \
       'T-1_C-O/C', 'rolling_7_days_avg', 'rolling_14_days_avg', \
       'rolling_21_days_avg', 'Popularity_log','pred_label_0', \
       'pred_label_1', 'pred_label_2']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

model_selection(X_train, X_test, y_train, y_test)

# final model - Random Forest
RF = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
RF.fit(X_train, y_train)
RF.score(X_test, y_test)

# predict 2019 data
X_pred = data_2019[['T-1_H-L/C', \
       'T-1_C-O/C', 'rolling_7_days_avg', 'rolling_14_days_avg', \
       'rolling_21_days_avg',  'Popularity_log','pred_label_0', \
       'pred_label_1', 'pred_label_2']]

# which stocks to buy based on model strategy
y_pred = RF.predict(X_pred)
data_2019['ylabel_pred'] = y_pred

# which stocks to buy based on sentiment score strategy
data_2019['stmt_pred'] = data_2019.apply(lambda x: function(x.pred_label_0, x.pred_label_2), axis = 1)

# export the data of year 2019 using information of year 2017 and 2018
data_2019.to_csv('data_2019_pred.csv')

# Third roll : Use data of 2017,2018,2019 to predict data of 2020.
# combine data of year 2017,2018,2019
data_17_19 = pd.concat([data_2017,data_2018,data_2019],ignore_index=True)

y = data_17_19.y_label
X = data_17_19[['T-1_H-L/C', \
       'T-1_C-O/C', 'rolling_7_days_avg', 'rolling_14_days_avg', \
       'rolling_21_days_avg', 'Popularity_log','pred_label_0', \
       'pred_label_1', 'pred_label_2']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

model_selection(X_train, X_test, y_train, y_test)

# final model - Random Forest
RF = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
RF.fit(X_train, y_train)
RF.score(X_test, y_test)

# predict 2020 data
X_pred = data_2020[['T-1_H-L/C', \
       'T-1_C-O/C', 'rolling_7_days_avg', 'rolling_14_days_avg', \
       'rolling_21_days_avg',  'Popularity_log','pred_label_0', \
       'pred_label_1', 'pred_label_2']]

# which stocks to buy based on model strategy
y_pred = RF.predict(X_pred)
data_2020['ylabel_pred'] = y_pred

# which stocks to buy based on sentiment score strategy
data_2020['stmt_pred'] = data_2020.apply(lambda x: function(x.pred_label_0, x.pred_label_2), axis = 1)

# export the data of year 2019 using information of year 2017 and 2018
data_2020.to_csv('data_2020_pred.csv')

#Fourth roll: Use data of year 2017,2018,2019,2020 to predict data of 2021.
# combine data of year 2017,2018,2019
data_17_20 = pd.concat([data_2017,data_2018,data_2019,data_2020],ignore_index=True)

y = data_17_20.y_label
X = data_17_20[['T-1_H-L/C', \
       'T-1_C-O/C', 'rolling_7_days_avg', 'rolling_14_days_avg', \
       'rolling_21_days_avg', 'Popularity_log','pred_label_0', \
       'pred_label_1', 'pred_label_2']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

model_selection(X_train, X_test, y_train, y_test)

# final model - Random Forest
RF = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
RF.fit(X_train, y_train)
RF.score(X_test, y_test)

# predict 2021 data
X_pred = data_2021[['T-1_H-L/C', \
       'T-1_C-O/C', 'rolling_7_days_avg', 'rolling_14_days_avg', \
       'rolling_21_days_avg', 'Popularity_log','pred_label_0', \
       'pred_label_1', 'pred_label_2']]

# which stocks to buy based on model strategy
y_pred = RF.predict(X_pred)
data_2021['ylabel_pred'] = y_pred

# which stocks to buy based on sentiment score strategy
data_2021['stmt_pred'] = data_2021.apply(lambda x: function(x.pred_label_0, x.pred_label_2), axis = 1)

# export the data of year 2019 using information of year 2017 and 2018
data_2021.to_csv('data_2021_pred.csv')