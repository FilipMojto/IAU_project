import pandas as pd
import scipy
import math
import matplotlib.pyplot as plt
import random as rn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, PowerTransformer
from matplotlib import pyplot
import seaborn as sns

columns_to_scale = ['pct_rage_click', 'pct_scroll_move']

power = PowerTransformer(
method='yeo-johnson',
standardize=True)
sess_train_df.loc[:, columns_to_scale] = power.fit_transform(sess_train_df[columns_to_scale])

pyplot.hist(sess_train_df[columns_to_scale], bins=25)