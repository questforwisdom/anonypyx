from anonypyx.generalisation.schema import GeneralisedSchema, build_column_groups

class HumanReadable(GeneralisedSchema):
    '''
    Generalisation schema with improved readability for humans.
    Does not support most operations.
    The generalised values are strings.
    '''
    @classmethod
    def create_for_data(cls, df, quasi_identifiers):
        categorical, integer, unaltered = build_column_groups(df, quasi_identifiers)
        return HumanReadable(categorical, integer, unaltered)

    @classmethod
    def from_json_dict(cls, json_dict):
        return HumanReadable(json_dict['categorical'], json_dict['integer'], json_dict['unaltered'])

    def __init__(self, categorical, integer, unaltered):
        '''
        Constructor.

        Parameters
        ----------
        categorical : list of str
            List of column names which are categorical quasi-identifiers.
        integer : list of str
            List of column names which are integer quasi-identifiers.
        unaltered : list of str
            List of column names which are not quasi-identifiers.
        '''
        super().__init__(unaltered)
        self._categorical = categorical
        self._integer = integer

    def to_json_dict(self):
        return {'categorical': self._categorical, 'integer': self._integer, 'unaltered': self._unaltered}

    def quasi_identifier(self):
        return self._integer + self._categorical

    def _generalise_partition(self, df):
        row = []
        for col in self._integer:
            row.append(self._to_string_interval(df[col]))

        for col in self._categorical:
            row.append(self._to_string_set(df[col]))

        return row

    def _to_string_interval(self, series):
        minimum = series.min()
        maximum = series.max()

        if maximum == minimum:
            return str(maximum)
        return f"{minimum}-{maximum}"

    def _to_string_set(self, series):
        l = [str(value) for value in series.unique()]
        l.sort()

        return ",".join(l)

