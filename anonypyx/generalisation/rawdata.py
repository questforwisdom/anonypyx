from anonypyx.generalisation import schema

import pandas as pd

class RawData(schema.GeneralisedSchema):
    '''
    Dummy generalised schema which does not alter the data other than removing duplicates and
    adding the count column. Can be used to process raw and generalised data in the samw way.
    '''
    @classmethod
    def create_for_data(cls, df, quasi_identifiers):
        categorical, integer, unaltered = schema.build_column_groups(df, df.columns)

        return RawData(categorical, integer, quasi_identifiers)

    @classmethod
    def from_json_dict(cls, json_dict):
        return RawData(json_dict['categorical'], json_dict['integer'], json_dict['quasi_identifier'])

    def __init__(self, categorical, integer, quasi_identifier):
        '''
        Constructor

        Parameters
        ----------
        categorical : list of str
            List of column names which have a categorical domain.
        integer : list of str
            List of column names which have a numerical domain.
        quasi_identifier : list of str
            List of column names which are quasi-identifiers.
        '''
        super().__init__(categorical + integer)
        self._quasi_identifier = quasi_identifier
        self._categorical = categorical
        self._integer = integer

    def to_json_dict(self):
        return {'categorical': self._categorical, 'integer': self._integer, 'quasi_identifier': self._quasi_identifier}

    def quasi_identifier(self):
        return self._quasi_identifier[:]

    def generalise(self, df, partitions):
        # dirty workaround, but serves its purpose:
        # overwrite the method to skip generalisation of
        # quasi-identifiers (the usual way of overwriting _generalise_partition()
        # is not possible here because the method must return exactly one 
        # generalised quasi-identifier per partition)
        partitions = [[i for p in partitions for i in p]]
        df = self._count_unique_unaltered_values(df, partitions)
        return df.drop('group_id', axis=1)

    def match(self, df, record, on):
        query = []
        for column in on:
            value = record[column]
            if isinstance(value, str):
                 query.append(f'`{column}` == "{value}"')
            else:
                query.append(f'`{column}` == {value}')

        query = ' and '.join(query)
        return df.query(query)

    def intersect(self, record_a, record_b, on, take_left, take_right):
        result = {}
        for column in on:
            if record_a[column] != record_b[column]:
                return None
            result[column] = record_a[column]

        self._copy_values(record_a, result, take_left)
        self._copy_values(record_b, result, take_right)

        return pd.Series(result)

    def _copy_values(self, origin, destination, columns):
        for column in columns:
            destination[column] = origin[column]

    def values_for(self, record, column):
        return {record[column]}

    def set_cardinality(self, record, on):
        return 1

    def select(self, df, query):
        df_query = []
        for col, value_range in query.items():
            if df[col].dtype.name == "category":
                subquery = []
                for value in value_range:
                    if isinstance(value, str):
                        subquery.append(f'`{col}` == "{value}"')
                    else:
                        subquery.append(f'`{col}` == {value}')
                df_query.append('(' + ' or '.join(subquery) + ')')
            else:
                df_query.append(f'`{col}` >= {value_range[0]}')
                df_query.append(f'`{col}` <= {value_range[1]}')


        df_query = ' and '.join(df_query)
        return df.query(df_query).index

    def query_overlap(self, record, query):
        for col, value_range in query.items():
            # TODO: checking pandas dtype does not work with series
            # the schema should now wheter the column is categorical or numerical
            # this is a workaround for now...
            if isinstance(value_range, set):
                if record[col] not in value_range:
                    return 0
            else:
                if record[col] < value_range[0] or record[col] > value_range[1]:
                    return 0
        return 1
