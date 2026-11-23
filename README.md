# Factors-Associated-with-Delayed-Medical-Care-Among-U.S.-Adults-
An Analysis of Hispanic Ethnicity, Communication Barriers, Insurance Coverage, Age and Geographic Region

######################################################################################################
# - Final Project: Logistic Regression for Predictors of Delayed Medical Care and Odds Ratio Graph - #
# -------------------------------------------------------------------------------------------------- #
#                         Karla-Aliya' Savon | karlaaliyasavon@usf.edu                               #
# --------------------------------------- 11/17/2026 ----------------------------------------------- #
# -------------------------------------------------------------------------------------------------- #
#                      (C) K-A'. Savon, 2026 | For Education and Research                            #
######################################################################################################

#Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import re
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools import add_constant
from pypdf import PdfReader

# Pull the dataset
df = pd.read_csv("meps.csv")

# Make it uppercase to lowercase
df.columns = df.columns.str.lower()

# Cleaning the variables
df = df[['dlayca42', 'hispanx', 'hwellspk', 'region24', 'age24x', 'inscov24']].copy()

# Remove the missing codes in MEPS
df = df[(df['dlayca42'] > 0) & (df['hispanx'] > 0) & (df['hwellspk'] > 0) & (df['region24'] > 0) & (df['age24x'] > 0) & (df['inscov24'] > 0)].copy()

# Turning the data into binary/ and renaming
    #Delayed Medical Care
df['delayed'] = (df['dlayca42'] == 1).astype(int)

    # Hispanic
df['Hispanic'] = (df['hispanx'] == 1).astype(int)

    # English or not (too large, have to tighten)
df['language_barrier'] = df['hwellspk'].isin([3,4]).astype(int)

    # Age
df['age'] = df['age24x'].astype(int)

    # Region (need to clarify the region areas in U.S.)
df['region'] = df['region24'].astype(int)

    # Insurance (need to clear up the different insurance types)
df['insurance'] = df['inscov24'].astype(int)

# Logistic Regression Formula/ what I'm testing for
model = smf.logit(
    "delayed ~ Hispanic * language_barrier + age + C(region) + C(insurance)",
    data=df
).fit()

# Print Model Summary
print(model.summary())

# To tell how many yeses and nos, n the different amounts for certain predictors
print(df['dlayca42'].value_counts())
print(df['hispanx'].value_counts())
print(df['hwellspk'].value_counts())
print(df['region24'].value_counts())
print(df['age24x'].value_counts())
print(df['insurance'].value_counts())

#Creating an Odds-Ratio Graph
or_values = np.exp(model.params)
plt.figure(figsize = (10,6))
plt.bar(or_values.index, or_values.values, color = 'blue',)
plt.axhline(y = 1, color = 'red', linestyle = '--')
plt.ylabel('Change in Odds of Delaying Medical Care')
plt.xlabel('Predictors')
plt.title('Factors Associated with Delayed Medical Care')
plt.xticks(rotation = 45)
plt.tight_layout()
plt.show()

#Logisitic Regression Results
<img width="1120" height="734" alt="image" src="https://github.com/user-attachments/assets/dc1ac5d0-6953-4617-8560-21ccdb00c324" />

#Odds Ratio Results
<img width="1000" height="600" alt="Factors Associated with Delayed Medical Care Odds Ratio Graph" src="https://github.com/user-attachments/assets/3bca9dec-fc06-4a39-a8cb-076ffc3639ab" />


