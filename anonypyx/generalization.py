import pandas as pd

def agg_categorical_column(series):
    # this is workaround for dtype bug of series
    series.astype("category")

    # l = [str(n) for n in set(series)]
    l = [str(n) for n in series.unique()]
    return ",".join(l)


def agg_numerical_column(series):
    minimum = series.min()
    maximum = series.max()
    if maximum == minimum:
        string = str(maximum)
    else:
        string = f"{minimum}-{maximum}"
    return string


def aggregate_partitions(df, partitions, feature_columns, sensitive_column):
    # TODO: inplace generalization to increase performance?
    aggregations = {}
    for column in feature_columns:
        if df[column].dtype.name == "category":
            aggregations[column] = agg_categorical_column
        else:
            aggregations[column] = agg_numerical_column
    unaltered_columns = df.columns.drop(feature_columns).to_list()
    result = pd.DataFrame()
    dfs = []
    for i, partition in enumerate(partitions):
        dfs.append(generalize_partition(df, partition, aggregations, unaltered_columns))
    return pd.concat(dfs)

def count_sensitive_values_in_partition(df, partition, unaltered_columns):
    if len(unaltered_columns) == 0:
        return pd.DataFrame([{'count': len(partition)}])

    counts = df.loc[partition].groupby(unaltered_columns, observed=True).size()
    return counts.reset_index(name="count")

def generalize_partition(df, partition, aggregations, unaltered_columns):
    values = df.loc[partition].agg(aggregations, squeeze=False)

    sensitive_counts = count_sensitive_values_in_partition(df, partition, unaltered_columns)
    result = pd.DataFrame()
 
    for column, value in values.items():
        sensitive_counts.loc[:, column] = value
    original_order = df.columns.to_list()
    original_order.append('count')
    return sensitive_counts[original_order]
 
