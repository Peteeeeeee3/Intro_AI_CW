import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
from IPython.display import display

#load dataset
intl_spi_data = pd.read_csv("spi_global_rankings_intl.csv")
#results_data = pd.read_csv("results.csv")

display(intl_spi_data.head())

qualified_teams = ["Qatar", "Germany", "Denmark", "Brazil", "France", "Belgium",
                   "Croatia", "Spain", "Serbia", "England", "Switzerland", "Netherlans",
                   "Argentina"]

intl_spi_data1 = os.path.join(".", "spi_global_rankings_intl.csv")
df = pd.read_csv(intl_spi_data1, na_values=['NA', '?'])

confederations = ["AFC", "CAF", "CONCACAF", "CONMEBOL", "OFC", "UEFA"]

# Strip non-numeric features from the dataframe
df = df.select_dtypes(include=['int', 'float'])

#print to check that this has worked

#collect the columns names for non-target features
result = []
for x in df.columns:
    if x != 'spi':
        result.append(x)
   
X = df[result].values
y = df['spi'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


from sklearn.linear_model import LinearRegression

model = LinearRegression()  
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
df_compare = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df_head = df_compare.head(25)
print(df_compare)