from scipy.stats import iqr  # Import the iqr function from scipy.stats
import pandas as pd

def identify_outliers(x):
    lower = x.quantile(0.25) - 1.5 * iqr(x)  # Use the imported iqr function
    upper = x.quantile(0.75) + 1.5 * iqr(x)

    return x[(x > upper) | (x < lower)]

PRODUCT_FILE_PATH = "../resources/product.csv"
USER_FILE_PATH = "../resources/user.csv"
SESSION_FILE_PATH = "../resources/session.csv"

# Read product, user, and session data into DataFrames
df_pro = pd.read_csv(PRODUCT_FILE_PATH, delimiter='\t')
df_usr = pd.read_csv(USER_FILE_PATH, delimiter='\t')
df_sess = pd.read_csv(SESSION_FILE_PATH, delimiter='\t')

outliers_pro_ean_usr = identify_outliers(df_pro["product_ean"])
print(outliers_pro_ean_usr)
# There are no outliers

outliers_sess_pad = identify_outliers(df_sess["page_activity_duration"])
print(outliers_sess_pad)
# There are no outliers

outliers_sess_rq = identify_outliers(df_sess["pct_rage_click"])
print(outliers_sess_rq)
# There are no outliers

outliers_sess_inp = identify_outliers(df_sess["pct_input"])
print(outliers_sess_inp)
# There are no outliers