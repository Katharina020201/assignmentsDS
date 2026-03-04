import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
import seaborn as sns
import statsmodels.api as sm


### Assignment B ###
#  Import the dataset births.csv and call this table births
births = pd.read_csv("births.csv")

# (a) Recode the variable child_birth into a new variable home where home=’at_home’ if the childbirth
# was a so-called first line child birth, at home, if not then home=’not_at_home’. Use for this
# the apply function with a lambda function or the map function with a dictionary, both are functions
# from the pandas library.
print(births.head())

line = "first line child birth, at home"
births['child_birth'] = births['child_birth'].map({line: 'at_home'}).fillna("not_at_home")
births = births.rename(columns={'child_birth': 'home'})
print(births.head())

# (b) Recode the variable parity in a new variable pari where pari has level primi if it concerns a first
# childbirth and multi if it is the second or more childbirth. You can do this again with apply or map
# function.

births['pari'] = births['parity'].apply(lambda x : 'primi' if x == 1 else 'multi')
births = births.drop('parity', axis=1)
print(births.head())
    
# (c) Recode the variable etnicity into a new variable etni where etni has level Dutch if the woman
# was Dutch and Not Dutch if she was not Dutch. Hint: use unique() of the pandas library to see
# which levels are in the variable etnicity.
print(births['etnicity'].unique())
births['etnicity'] = births['etnicity'].map({'Dutch': 'Dutch'}).fillna("Not Dutch")
births = births.rename(columns={'etnicity': 'etni'})
print(births.head())

# (d) Using the sklearn library make a logistic regression model with the function LogisticRegression
# for the probability of childbirth at home with the variables pari, age_cat (= age categorised), etni and urban (urbanisation degree). 
# View the outcomes from the model with the classification_report() function.

# for col in births.columns:
#     print(f"Unique values in {col}: {births[col].unique()}")

# Recode categorical variables to numeric (dummy variables)
births['home'] = births['home'].map({'at_home': 1, 'not_at_home': 0})
births['urban'] = births['urban'].map({
    'very strong' : 0,
    'strong': 1,
    'moderate': 2,
    'low': 3,
    'not': 4
})
births['age_cat'] = births['age_cat'].map({
    '25-29 year' : 0,
    '30-34 year' : 1,
    '> 35 year' : 2,
    '< 25 year' : 3
})
births['pari'] = births['pari'].map({'primi': 1, 'multi': 0})
births['etni'] = births['etni'].map({'Dutch': 1, 'Not Dutch': 0})


Y = births['home'] # Dependent variable
X = births[['urban', 'age_cat', 'etni', 'pari']] # Independent variable

# split X and y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=16)

print("Logistic Regression")
model = LogisticRegression(random_state=16)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# (e) Using the same sklearn library make a decision tree for the probability of childbirth at home with
# the same variables as in the logistic regression model. View the decision tree with the function
# tree.plot_tree from package tree.
print("Decision Tree")
decision_tree = DecisionTreeClassifier(criterion='gini', max_depth=None)
decision_tree.fit(X_train, y_train)
y_pred_tree = decision_tree.predict(X_test)
tree.plot_tree(decision_tree)
plt.show()
print(classification_report(y_test, y_pred_tree))

# depth = 8 ?

# (f) For assessing which model, the logistic regression model or the decision tree, fits better the data we
# should fit the models on a training set and calculate accuracy statistics on a test set (or use cross
# validation). For this we use sklearn.model_selection and cross_val_score. Which model fits
# the data better?


# Results are almost identical for both the regression model and the decision tree 
# when using different test sizes and cross validation attempts
# The decision tree seems to perform slightly higher with cross validation scores, but this does not seem significant

# model, decision_tree
print(cross_val_score(model, X, Y))
print(cross_val_score(decision_tree, X, Y))

# 25%
# [0.65246258 0.65354901 0.65684795]
# [0.6551183  0.65704973 0.65805517]
# 20%
# [0.65246258 0.65354901 0.65684795]
# [0.6551183  0.65704973 0.65805517]
# [0.6517453  0.65415954 0.65315361 0.65251509 0.65603622]
# [0.65737853 0.65536666 0.65697616 0.65895372 0.65875252]
# [0.65318687 0.65173829 0.65463544 0.65560116 0.65306615 0.6560425 ]
# [0.65898117 0.65210043 0.65391115 0.66018831 0.65789474 0.65821562]
# 10%
# [0.65246258 0.65354901 0.65684795]
# [0.6551183  0.65704973 0.65805517]
