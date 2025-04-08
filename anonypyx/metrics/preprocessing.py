import anonypyx.generalisation

import pandas as pd

def preprocess_prediction(prediction_df):
    """
    Prepares the adversary's prediction for most privacy metrics.

    Parameters
    ----------
    prediction_df : pd.DataFrame
        The adversary's prediction. It must contain the unique identifier of the targets in a 
        column ID and then one column for every distinct value of the sensitive attribute (names
        must match the values) which contains the adversary's confidence in the
        corresponding value for the given individual. These confidence values do not have to be
        normalised (i.e. rows such as (ID=0, S_1=0, S_2=1, S_3=2, S_4=5) may be used here). 
        There must be exactly one record/row per targeted individual.

    Returns
    -------
        A data frame where the ID column is used as an index and the confidence levels are
        normalised in the range [0, 1].
    """
    prediction_df = prediction_df.set_index('ID')
    prediction_df = prediction_df.div(prediction_df.sum(axis=1), axis=0)
    return prediction_df

def preprocess_original_data_for_privacy(original_df):
    """
    Prepares the original input data frame for most privacy metrics.

    Parameters
    ----------
    original_df : pd.DataFrame
        The original data frame before anonymisation. If multiple data sets are released, this
        must contain all records, i.e. provide the union of all original data sets instead. It 
        must contain a column ID assigning a unique identifier to each targeted individual.
        IDs must start at 0 and increase by 1 for every target.
        There must be exactly one record/row per targeted individual.

    Returns
    -------
        A data frame where the ID column is used as an index.
    """
    return original_df.set_index('ID')

class PreparedUtilityDataFrame:
    """
    This class represents a pandas.DataFrame enriched with some preprocessed
    metadata used by the utility metrics.
    """
    def __init__(self, df, schema, quasi_identifier):
        """
        Constructor.

        Parameters
        ----------
        df : pd.DataFrame
            The data frame for which utility will be measured.
            Note: It may be altered by this class and its method. Pass
            a copy if you want to keep the original data frame.
        schema : anonypyx.generalisation.GeneralisationStrategy
            The generalisation strategy/schema used by df.
        quasi_identifier : list of str
            The names of the columns from the original data frame which serve as 
            a quasi-identifier (i.e. column names before generalisation).
        """
        self._df = df
        self._original_quasi_identifier = quasi_identifier
        self._schema = schema
        self._group_sizes = []

        self._preprocess_groups()

    def _preprocess_groups(self):
        equivalence_classes = self._df.groupby(by=self._schema.quasi_identifier(), observed=True)
        group_id = 0
        self._df['group_id'] = -1
        for _, group_df in equivalence_classes:
            self._group_sizes.append(group_df['count'].sum())
            for i in group_df.index:
                self._df.at[i, 'group_id'] = group_id
            group_id += 1

    def df(self):
        """
        Returns a reference to the data frame. The data frame contains an additional column 'group_id'
        which provides a unique identifier for every equivalence class.
        """
        return self._df

    def schema(self):
        """
        Returns a reference to the generalisation schema.
        """
        return self._schema

    def original_quasi_identifier(self):
        """
        Returns a list containing the original column names serving as a quasi-identifier.
        """
        return self._original_quasi_identifier

    def group_size(self, group_id):
        """
        Returns the size of the equivalence class with the given group_id, i.e. the number
        of data points contained in this class.
        """
        return self._group_sizes[group_id]

    def num_groups(self):
        return len(self._group_sizes)

    @classmethod
    def from_raw_data(cls, df, quasi_identifier):
        """
        Prepares the original input data frame for most utility metrics.
    
        Parameters
        ----------
        original_df : pd.DataFrame
            The original data frame before anonymisation. If multiple data sets are released, this
            must contain all records, i.e. provide the union of all original data sets instead. It 
            must contain a column ID assigning a unique identifier to each targeted individual.
            IDs must start at 0 and increase by 1 for every target.
            There must be exactly one record/row per targeted individual.
        quasi_identifier : list of str
            The names of the columns which serve as a quasi-identifier.
        
        Returns
        -------
        The PreparedUtilityDataFrame for the given data frame.
        """
        df = df.drop('ID', axis=1)
        categorical = [c for c in df.columns if df[c].dtype.name == 'category']
        numerical = [c for c in df.columns if df[c].dtype.name != 'category']
        schema = anonypyx.generalisation.RawData(categorical, numerical, quasi_identifier)
        df = schema.generalise(df, [[i] for i in df.index])
        return PreparedUtilityDataFrame(df, schema, quasi_identifier)

