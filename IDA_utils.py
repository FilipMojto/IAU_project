
def split_data(df, split = 20):
    num_rows_first_subset = int(len(df) * (split / 100.0))

    first_subset = df.head(num_rows_first_subset).copy()
    second_subset = df.tail(len(df) - num_rows_first_subset).copy()

    return first_subset, second_subset