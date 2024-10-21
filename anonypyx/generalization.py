import pandas as pd

class Interval:
    def generalise(self, series):
        return [series.min(), series.max()]

    def new_names(self, column):
        return [column + '_min', column + '_max']

class OneHot:
    def generalise(self, series):
        return [series.max()]

    def new_names(self, column):
        return [column]

class HumanReadableInterval:
    def generalise(self, series):
        minimum = series.min()
        maximum = series.max()

        if maximum == minimum:
            return [str(maximum)]
        else:
            return [f"{minimum}-{maximum}"]

    def new_names(self, column):
        return [column]

class HumanReadableSet:
    def generalise(self, series):
        l = [str(value) for value in series.unique()]
        l.sort()

        return [",".join(l)]

    def new_names(self, column):
        return [column]

def aggregate_partitions(df, partitions, aggregations):
    # TODO: if no selection is present, use a default config
    unaltered_columns = df.columns.drop(aggregations.keys()).to_list()
    new_data = []
    for i, partition in enumerate(partitions):
        new_data += generalize_partition(df, partition, aggregations, unaltered_columns)

    new_columns = []
    for column in df.columns:
        if column in unaltered_columns:
            new_columns.append(column)
        else:
            new_columns += aggregations[column].new_names(column)
    new_columns.append('count')
    return pd.DataFrame(new_data, columns=new_columns)

def count_sensitive_values_in_partition(df, partition, unaltered_columns):
    if len(unaltered_columns) == 0:
        return pd.DataFrame([{'count': len(partition)}])

    counts = df.loc[partition].groupby(unaltered_columns, observed=True).size()
    return counts.reset_index(name="count")

def generalize_partition(df, partition, aggregations, unaltered_columns):
    sensitive_counts = count_sensitive_values_in_partition(df, partition, unaltered_columns)
    generalised_descriptions = {column: generaliser.generalise(df.loc[partition][column]) for column, generaliser in aggregations.items()}
    result = []

    for i in range(len(sensitive_counts.index)):
        row = []
        for column in df.columns:
            if column in generalised_descriptions.keys():
                row += generalised_descriptions[column]
            else:
                row.append(sensitive_counts[column][i])
        
        row.append(sensitive_counts['count'][i])
        result.append(row)

    return result

