import pandas as pd

def build_column_groups(df, quasi_identifiers):
    categorical = []
    integer = []
    unaltered = []

    for column in quasi_identifiers:
        if df[column].dtype.name == "category":
            categorical.append(column)
        else:
            integer.append(column)

    altered_columns = categorical + integer
    unaltered = df.columns.drop(altered_columns).to_list()

    return categorical, integer, unaltered

class GeneralisedSchema:
    '''
    Abstract base class for generalised data schemas.
    A schema defines how the generalised data is used.
    '''
    @classmethod
    def create_for_data(cls, df, quasi_identifiers):
        '''
        Returns a GeneralisedSchema instance for the given data frame where
        the given quasi identifiers have been generalised.

        Parameters
        ----------
        df : pandas.DataFrame
            The data frame from which the schema is derived.
        quasi_identifiers : list of str
             The names of the columns from the data frame which will be generalised.
        '''
        raise NotImplementedError()

    def __init__(self, unaltered_columns):
        '''
        Constructor. Pass the list of column names which are not generalised
        as an argument.
        '''
        self._unaltered = unaltered_columns

    def generalise(self, df, partitions):
        '''
        Generalises the given data frame according to the given partitioning.

        Parameters
        ----------
        df : pandas.DataFrame
            The data frame to generalise. Its column must match those provided when
            creating the instance from which this method is called.
        partitions : list of pandas indices
            Each index in the list defines a subset of rows from the data frame
            which will be generalised. The subsets must not overlap.

        Returns
        -------
        A pandas.DataFrame which has been generalised according to this schema.
        '''
        df = self._preprocess(df)

        part_1 = self._generalise_quasi_identifiers(df, partitions)
        part_2 = self._count_unique_unaltered_values(df, partitions)

        return part_1.merge(part_2, on='group_id')

    def match(self, df, record, on):
        '''
        Checks which rows in the dataframe are consistent with the
        given record when projected to the given columns. A row is 
        consistent with the record if they overlap, i.e. if there is
        at least one potential input row which is consistent with
        both generalised descriptions.

        Parameters
        ----------
        df : pandas.DataFrame
            The data frame which is checked for overlaps. It must be
            generalised according to this schema.
        record : pandas.Series or dict-like
            The record which is checked for overlaps. It must be 
            generalised according to this schema. If this is a
            pandas.Series, it must be indexed by the column names.
        on : list of str
            The column names from the original data which are checked 
            for overlaps. Columns not provided in this list are ignored.

        Returns
        -------
        An index of the given data frame containing all matching rows.

        '''
        raise NotImplementedError()

    def intersect(self, record_a, record_b, on, take_left, take_right):
        '''
        Returns the intersection of two generalised records.

        Parameters
        ----------
        record_a : pandas.Series or dict-like
            The first record/row to intersect. It must be generalised
            according to this schema.
        record_b : pandas.Series or dict-like
            The second record/row to intersect. It must be generalised
            according to this schema.
        on : list of str
            The column names from the original data for which the intersection 
            is computed. Columns not provided in this list are ignored.
        take_left : list of str
            The column names from the original data for which the value from
            record_a is copied into the resulting record. These columns
            must not appear in on or take_right.
        take_lright : list of str
            The column names from the original data for which the value from
            record_b is copied into the resulting record. These columns
            must not appear in on or take_left.

        Returns
        -------
        The record/row representing the intersection of both records.
        If the intersection is empty, None is returned instead.
        '''
        raise NotImplementedError()

    def values_for(self, record, column):
        '''
        Returns the generalised value set of a record for the given column.

        Parameters
        ----------
        record : pandas.Series or dict-like
            The record for which the value set is retrieved.
        column : str
            The name of the original column for which the value set is retrieved.

        Returns
        -------
        A set containing the values which correspond to the generalised value
        for the given column in the given record.
        '''
        raise NotImplementedError()

    def _preprocess(self, df):
        '''
        Overwrite this method in subclasses.
        It receives the raw data frame as input and must return a data frame
        containing all columns required by the generalisation schema.
        '''
        return df

    def quasi_identifier(self):
        '''
        Overwrite this method in subclasses.
        It must return the list of column names corresponding to the new quasi-identifiers.
        Ensure that the order is the same as returned by _generalise_partition()!
        '''
        raise NotImplementedError()

    def _generalise_partition(self, df):
        '''
        Overwrite this method in subclasses.
        It receives a pandas data frame containing the rows from a single
        partition (see generalise()). It must return a list of cell values
        containing the generalised quasi-identifier values corresponding
        to this partition in the same order as returned by quasi_identifier().
        '''
        raise NotImplementedError()

    def _generalise_quasi_identifiers(self, df, partitions):
        data = []
        columns = self.quasi_identifier()
        for i, partition in enumerate(partitions):
            row = self._generalise_partition(df.loc[partition])
            row.append(i)
            data.append(row)
        columns.append('group_id')

        return pd.DataFrame(data, columns=columns)

    def _count_unique_unaltered_values(self, df, partitions):
        unaltered_data = []

        for i, partition in enumerate(partitions):
            sensitive_counts = count_sensitive_values_in_partition(df, partition, self._unaltered)

            for j in range(len(sensitive_counts.index)):
                row = []
                for column in self._unaltered:
                    row.append(sensitive_counts[column][j])
                row.append(sensitive_counts['count'][j])
                row.append(i)
                unaltered_data.append(row)

        return pd.DataFrame(unaltered_data, columns=self._unaltered + ['count', 'group_id'])

    def is_original_column(self, column):
        '''
        Returns True if and only if the given column name is an unaltered column.
        Returns False else.
        '''
        return column in self._unaltered

def count_sensitive_values_in_partition(df, partition, unaltered_columns):
    if len(unaltered_columns) == 0:
        return pd.DataFrame([{'count': len(partition)}])

    counts = df.loc[partition].groupby(unaltered_columns, observed=True).size()
    return counts.reset_index(name="count")

