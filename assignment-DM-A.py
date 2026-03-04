import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
import numpy as np
import seaborn as sns
import statsmodels.api as sm
# import statsmodels.formula.api as sfm


### Assigment A ###
#  Import the data from voorbeeld7_1.sav and save the table under the name chol1
chol1 = pd.read_spss('voorbeeld7_1.sav') # Returns a dataframe

chol1.info()
print(chol1.head())
print(chol1['sekse'].unique())
print(chol1['alcohol'].unique())

# (a) Make a scatter plot, with the function lmplot (using the seaborn library), 
# from the column cholesterol chol (y-axis) and age leeftijd (x-axis). 
# Add then a regression line to the graph with lmplot(..,fit_reg=True).
sns.lmplot(data=chol1, x='leeftijd', y="chol", fit_reg=True)
plt.show()

# (b) Fit a linear model for chol with leeftijd using the function ols (using the statsmodels library). The
# formula for the model is chol˜leeftijd. Save the fitobject under the name fit1. View the result with fit1.summary().

Y = chol1['leeftijd'] # Dependent variable
X = chol1['chol'] # Independent variable
X = sm.add_constant(X)
model = sm.OLS(Y,X)
fit1 = model.fit() # fitobject
print(fit1.summary())


# (c) Fit a model fit2 for chol with leeftijd, bmi, sekse and alcohol. Which factors are statistically significant?

# Statistically significant factors are those with a p-value less than 0.05 -> P>|t| 
# => All variables are statistically significant 


# With statsmodels.formula.api
# print("With statsmodels.formula.api")
# model3 = sfm.ols("chol ~ C(sekse, Treatment('man')) + leeftijd + bmi + C(alcohol, Treatment('niet drinkers'))", data=chol1).fit()
# print(model3.summary())

# With dummies
print("Without statsmodels.formula.api")
# chol1 = pd.get_dummies(chol1, columns=['sekse', 'alcohol'], drop_first=True, dtype=float)

# chol1.info()
# print(chol1.head())

# Y = chol1['chol'] # Dependent variable
# X = chol1[['leeftijd', 'bmi', 'sekse_vrouw', 'alcohol_niet drinkers', 'alcohol_zware drinkers']] # Independent variable

# Self defined
chol1['sekse_num'] = chol1['sekse'].map({'man': 0, 'vrouw': 1})
chol1['alcohol_num'] = chol1['alcohol'].map({
    'niet drinkers': 1,
    'matige drinkers': 2,
    'zware drinkers': 3
})

chol1.info()
print(chol1.head())

Y = chol1['chol'] # Dependent variable
X = chol1[['leeftijd', 'bmi', 'sekse_num', 'alcohol_num']] # Independent variable

X = sm.add_constant(X)
model2 = sm.OLS(Y,X)
fit2 = model2.fit() # fitobject
print(fit2.summary())
print(fit2.pvalues)

# (d) Add the residuals from the model fit2 to the table chol1 and make a histogram from the residuals.
chol1['residuals'] = fit2.resid
sns.histplot(chol1['residuals'], bins=10, kde=True)
plt.hist(chol1['residuals'], bins=10)
plt.show()
