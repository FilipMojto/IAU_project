import pandas as pd
import math
from typing import List, Dict, Literal

def num_to_category(df: pd.DataFrame, col: str, num_bins: int = None):

    if not num_bins:
        num_bins = math.floor(df[col].max())        

    df[col] = pd.cut(df[col], bins=num_bins, labels=False)
    df.drop(col, axis=1)

def timestamp_to_category(df: pd.DataFrame, col: str, mode: Literal['day', 'month', 'year'] = 'month'):
    df[col] = pd.to_datetime(df[col])
    # Extract month and create a new column 'month'

    if mode == 'day':
        df[col] = df[col].dt.day
    elif mode == 'month':
        df[col] = df[col].dt.month
    elif mode == 'year':
        df[col] = df[col].dt.year


def split_data(df: pd.DataFrame, proportion: float = 0.8):

    if not 0 <= proportion <= 1:
        raise ValueError("Invalid proportion value, valid is within 0 and 1!")
    #print("ERE!")
    num_rows_first_subset = int(len(df) * proportion)

    train_df = df.head(num_rows_first_subset).copy()
    test_df = df.tail(len(df) - num_rows_first_subset).copy()

    return train_df, test_df


def process_missing_vals(dataframe, limit=1):
    print(f'\nstarting the process...')

    #Here we calculate the actual limit for NaNs which can be dropped
    limit_count = math.ceil((len(dataframe) / 100) * limit)
    print(f'Limit: {limit_count}')
    
    for col in dataframe.columns:
        na_count_before = dataframe[col].isna().sum()

        #If there are no NaNs in the current column, we move to the next one
        if na_count_before == 0:
            print(f"Column '{col}': No NaN values initially.")
            continue

        #If the count of NaNs is acceptable, we remove the rows containing them
        if na_count_before <= limit_count:
            print(f"Column '{col}': Dropping NaN values...")
            dataframe.dropna(subset=[col], inplace=True)
        #else we apply different strategies for numeric and non-numeric data
        else:
            print(f"Column '{col}': Too many NaN values...", end=" ")
            is_numeric = pd.to_numeric(dataframe[col], errors='coerce').notna().all()
            
            if not is_numeric:
                print("Imputing the most common value...")
                most_common_value = dataframe[col].mode()[0]  # Get the most common value
                dataframe[col].fillna(most_common_value, inplace=True)
            else:
                print("Imputing the median value...)")
                median = dataframe[col].median()
                dataframe[col].fillna(median, inplace=True)    
        
        na_count_after = dataframe[col].isna().sum()
        print(f"NaN count before - {na_count_before}, after - {na_count_after}")

    print(f'Process done!\n')

def process_outliers(df, limit = 1, threshold=3):
    print("\nStarting process...")

    for col in df.columns:
        is_numeric = pd.to_numeric(df[col], errors='coerce').notna().all()

        if not is_numeric:
            print(f"Column '{col}': Not numerical! Skipping...")
            continue

        z_scores = (df[col] - df[col].mean()) / df[col].std()
        outliers_mask = abs(z_scores) > threshold
        outliers_c = outliers_mask.sum()
        print(f"Column '{col}': Detected {outliers_c} outliers! Processing...", end=" ")
        
        if outliers_c <= math.ceil((len(df)/100) * limit):
            print(f"Removing outliers...")
            df = df.loc[~outliers_mask]
        else:
            print(f"Replacing with a limit value...")
            p5 = df[col].quantile(0.05)
            p95 = df[col].quantile(0.95)

            p5 = df[col].quantile(0.05)
            p95 = df[col].quantile(0.95)
            
            df.loc[df[col] > p95, col] = p95
            df.loc[df[col] < p5, col] = p5
            
        print(f"Remaining rows after removing outliers: {len(df)}")

    print("Ending process...\n")


def frequency_tables(df: pd.DataFrame, target_column: str, predictors: tuple[str]):
    if len(predictors) == 0:
        raise ValueError("At least one predictor needed!")

    result = {}

    for predictor in predictors:
        freq_table = pd.crosstab(df[predictor], df[target_column])
        result[predictor] = freq_table

    return result


def one_rule_alg(freq_tables: Dict[str, pd.DataFrame], metric: Literal['accuracy', 'precision', 'recall'] = 'accuracy'):
    accuracies = {}
    rules = {}

    for table in freq_tables:
        positive = [0, 0]
        negative = [0, 0]

        for index, row in freq_tables[table].iterrows():

            row_vals = row.items()
            row_vals = list(row_vals)      
            if row_vals[0][1] > row_vals[1][1]:
                rules[table] = freq_tables[table].columns[0]
                positive[0] +=  row_vals[0][1]
                positive[1] += row_vals[1][1]
            else:
                rules[table] = freq_tables[table].columns[1]
                negative[0] +=  row_vals[0][1]
                negative[1] += row_vals[1][1]
        
        try:
            if metric == 'accuracy':
                accuracies[table] = ((positive[0] + negative[1])/(positive[0] + positive[1] + negative[0] + negative[1]))
            elif metric == 'precision':
                accuracies[table] = ((positive[0])/(positive[0] + positive[1]))
            elif metric == 'recall':
                accuracies[table] = ((positive[0])/(positive[0] + negative[0]))
        except ZeroDivisionError:
            accuracies[table] = -1
    

            
    return rules, accuracies, {'used_metric': metric}



