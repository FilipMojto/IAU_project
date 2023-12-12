import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# from IDA_utils import *
from scipy.stats import shapiro, kstest
from typing import Literal
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

PRODUCT_FILE_PATH = "resources/product.csv"
USER_FILE_PATH = "resources/user.csv"
SESSION_FILE_PATH = "resources/session.csv"

prod_df = pd.read_csv(PRODUCT_FILE_PATH, delimiter='\t')
user_df = pd.read_csv(USER_FILE_PATH, delimiter='\t')
sess_df = pd.read_csv(SESSION_FILE_PATH, delimiter='\t')

process_missing_vals(prod_df)
process_missing_vals(user_df)
process_missing_vals(sess_df)

process_outliers(prod_df)
process_outliers(user_df)
process_outliers(sess_df)

prod_train_df, prod_test_df = split_data(prod_df, 0.8)
user_train_df, user_test_df = split_data(user_df, 0.8)
sess_train_df, sess_test_df = split_data(sess_df, 0.8)

dt_classifier = DecisionTreeClassifier()

sess_train_df_out = sess_train_df['ack']
sess_test_df_out = sess_test_df['ack']

tren_model = dt_classifier.fit(sess_train_df, sess_train_df_out )
predicted_labels = tren_model.predict(sess_train_df)

print('Training accuracy is : ',tren_model.score(sess_train_df,sess_train_df_out))
print('Test Accuracy is : ',tren_model.score(sess_test_df, sess_test_df_out))