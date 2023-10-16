
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
    rows = []
    for i, partition in enumerate(partitions):
        rows += generalize_partition(df, partition, aggregations, sensitive_column)
    return rows

def count_sensitive_values_in_partition(df, partition, sensitive_column):
    return df.loc[partition].groupby(sensitive_column, observed=True).agg({sensitive_column: "count"})

def generalize_partition(df, partition, aggregations, sensitive_column):
    rows = []
    # partition -> generalized quasi-identifier + sensitive value distribution
    values = df.loc[partition].agg(aggregations, squeeze=False)
    sensitive_counts = count_sensitive_values_in_partition(df, partition, sensitive_column)
    for sensitive_value, count in sensitive_counts[sensitive_column].items():
        # if count == 0:
        #     continue
        values[sensitive_column] = sensitive_value
        values["count"] = count
        rows.append(values.copy())
    return rows
