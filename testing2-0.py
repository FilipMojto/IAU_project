import pandas as pd
import scipy
import math
import matplotlib.pyplot as plt
import random as rn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, PowerTransformer
from matplotlib import pyplot
import seaborn as sns

# Read the CSV file
sess_df = pd.read_csv('fixed_sess.csv')

# Only num columns for correlation
numeric_columns = sess_df.select_dtypes(include=['float64', 'int64']).columns
cor_matrix_pred = sess_df[numeric_columns].corr()

# Vytvorenie heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(cor_matrix_pred, annot=True, cmap='coolwarm', fmt='.3f')
plt.title('Korelácie medzi numerickými atribútmi k predikovanej premennej ack pred transformáciou')
plt.show()
