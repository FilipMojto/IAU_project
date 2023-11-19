import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
# Define file paths for data
PRODUCT_FILE_PATH = "../resources/product.csv"
USER_FILE_PATH = "../resources/user.csv"
SESSION_FILE_PATH = "../resources/session.csv"

# Read product, user, and session data into DataFrames
df_pro = pd.read_csv(PRODUCT_FILE_PATH, delimiter='\t')
df_usr = pd.read_csv(USER_FILE_PATH, delimiter='\t')
df_sess = pd.read_csv(SESSION_FILE_PATH, delimiter='\t')

print(df_pro.info())
#print(df_usr.info())
#print(df_sess.info())

# Plot histograms for numerical features
numerical_features = ['product_ean']
for feature in numerical_features:
    plt.figure(figsize=(8, 6))
    sns.histplot(df_pro[feature], kde=True)
    plt.title(f'Distribution of {feature}')
    plt.show()

# Plot bar plots for categorical features
categorical_features = ['location']
for feature in categorical_features:
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df_pro, x=feature)
    plt.title(f'Counts of {feature}')
    plt.xticks(rotation=45)
    plt.show()