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

spi = pd.read_csv("spi_global_rankings_intl.csv")
spi = spi.drop(columns=['confed'])

results = pd.read_csv("results.csv")

results = results.drop(columns=['date', 'tournament', 'city', 'country', 'neutral'])

home_only = results.merge(spi, left_on='home_team', right_on='name')
print(home_only)

final_merge = home_only.merge(spi, left_on='away_team', right_on='name')
print(final_merge)

final_merge.drop(columns=['name_x', 'name_y'])

final_merge.to_excel('merging_test.xlsx', sheet_name='teams, spi & scores')