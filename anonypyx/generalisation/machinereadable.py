from anonypyx.generalisation.schema import GeneralisedSchema, build_column_groups

import pandas as pd

class MachineReadable(GeneralisedSchema):
    '''
    Generalised schema which is machine readable.
    Integer columns are generalised to intervals with a minimum and maximum
    column. Categorical columns are replaced by one boolean column for each
    value in its domain such that True indicates that the value appears in a
    partition (conceptually similar to a one-hot vector).
    '''
    @classmethod
    def create_for_data(cls, df, quasi_identifiers):
        categorical, integer, unaltered = build_column_groups(df, quasi_identifiers)

        for column in categorical:
            value_columns = [column + '_' + value for value in df[column].unique()]

        
        for column in integer:
            df[column + '_min'] = df[column]
            df[column + '_max'] = df[column]

        one_hot_sets = {col: [col + '_' + val for val in df[col].unique()] for col in categorical}
        intervals = {col: (col + '_min', col + '_max') for col in integer}

        return MachineReadable(one_hot_sets, intervals, unaltered)

    def __init__(self, one_hot_sets, intervals, unaltered):
        '''
        Constructor

        Parameters
        ----------
        one_hot_sets : dict mapping str to list of str
            The original column names of categorical quasi-identifiers are the keys.
            They are mapped to the list of their corresponding "one-hot" column names.
        intervals : dict mapping str to (str, str)
            The original column names of integer quasi-identifiers are the keys.
            They are mapped to tuples containing the corresponding minimum and maximum 
            column (in that order)
        unaltered : list of str
            List of column names which are not quasi-identifiers.
        '''
        super().__init__(unaltered)
        self._one_hot_sets = one_hot_sets
        self._intervals = intervals

    def _preprocess(self, df):
        for column, interval in self._intervals.items():
            df[interval[0]] = df[column]
            df[interval[1]] = df[column]
        df = df.drop(self._intervals.keys(), axis=1)
        df = pd.get_dummies(df, columns=self._one_hot_sets.keys(), prefix=self._one_hot_sets.keys())

        for one_hot_set in self._one_hot_sets.values():
            for column in one_hot_set:
                if column not in df.columns:
                    df[column] = False

        return df

    def quasi_identifier(self):
        qi = []
        for interval in self._intervals.values():
            qi.append(interval[0])
            qi.append(interval[1])

        for one_hot_set in self._one_hot_sets.values():
            for one_hot_col in one_hot_set:
                qi.append(one_hot_col)

        return qi

    def _generalise_partition(self, df):
        row = []
        for interval in self._intervals.values():
            row.append(df[interval[0]].min())
            row.append(df[interval[1]].max())

        for one_hot_set in self._one_hot_sets.values():
            for one_hot_col in one_hot_set:
                row.append(df[one_hot_col].max())

        return row

    def match(self, df, record, on):
        df_filter = True
        for column in on:
            column_filter = False
            if column in self._intervals:
                min_col, max_col = self._intervals[column]
                column_filter = (df[min_col] <= record[max_col]) & (df[max_col] >= record[min_col])
            elif column in self._one_hot_sets:
                for value_column in self._one_hot_sets[column]:
                    if record[value_column] == 1:
                        column_filter = column_filter | (df[value_column] == 1)
            else:
                column_filter = df[column] == record[column]
            df_filter = df_filter & column_filter

        return df[df_filter].index

    def intersect(self, record_a, record_b, on, take_left, take_right):
        result = {}
        for column in on:
            if column in self._intervals:
                min_col, max_col = self._intervals[column]
                result[min_col] = max(record_a[min_col], record_b[min_col])
                result[max_col] = min(record_a[max_col], record_b[max_col])

                if result[min_col] > result[max_col]:
                    return None

            elif column in self._one_hot_sets:
                at_least_one = False
                for value_column in self._one_hot_sets[column]:
                    result[value_column] = min(record_a[value_column], record_b[value_column])
                    at_least_one = at_least_one or result[value_column]

                if not at_least_one:
                    return None

            else:
                if record_a[column] != record_b[column]:
                    return None
                result[column] = record_a[column]

        self._copy_values(record_a, result, take_left)
        self._copy_values(record_b, result, take_right)

        return pd.Series(result)

    def _copy_values(self, origin, destination, columns):
        for column in columns:
            if column in self._intervals:
                min_col, max_col = self._intervals[column]
                destination[min_col] = origin[min_col]
                destination[max_col] = origin[max_col]
            elif column in self._one_hot_sets:
                for value_column in self._one_hot_sets[column]:
                    destination[value_column] = origin[value_column]
            else:
                destination[column] = origin[column]

    def values_for(self, record, column):
        if column in self._unaltered:
            return {record[column]}
        if column in self._intervals:
            min_col, max_col = self._intervals[column]
            return set(range(record[min_col], record[max_col] + 1))
        # TODO: workaround, dependency on naming scheme of value columns
        return {c.removeprefix(column + '_') for c in self._one_hot_sets[column] if record[c]}

