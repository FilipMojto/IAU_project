import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
# Define file paths for data
PRODUCT_FILE_PATH = "resources/product.csv"
USER_FILE_PATH = "resources/user.csv"
SESSION_FILE_PATH = "resources/session.csv"

# Read product, user, and session data into DataFrames
df_pro = pd.read_csv(PRODUCT_FILE_PATH, delimiter='\t')
df_usr = pd.read_csv(USER_FILE_PATH, delimiter='\t')
df_sess = pd.read_csv(SESSION_FILE_PATH, delimiter='\t')

duplikaty = df_pro.duplicated()

print(duplikaty.any())
print(df_pro[duplikaty])
df_pro.drop_duplicates(inplace=True)
print(df_pro.shape)

duplikaty2 = df_usr.duplicated()

print(duplikaty2.any())
print(df_usr[duplikaty2])
df_usr.drop_duplicates(inplace=True)
print(df_usr.shape)

duplikaty3 = df_sess.duplicated()

print(duplikaty3.any())
print(df_sess[duplikaty3])
df_sess.drop_duplicates(inplace=True)
print(df_sess.shape)

df_duplicates = df_sess[df_sess.duplicated(keep=False)]
print(df_duplicates)

print("===================")

print(df_pro.isnull().sum())

print(df_usr.isnull().sum())

print(df_sess.isnull().sum())