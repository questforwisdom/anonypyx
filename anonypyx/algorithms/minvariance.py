
class MInvariance:
    def __init__(self, m, last_df=None, last_partition=None):
        '''
        Constructor.

        Parameters
        ----------

        m : int
            Parameter m from m-invariance. Determines the number of different sensitive
            values which must be present in each equivalence class.
        last_df : pandas.DataFrame
            Data frame which was the input of the last call to partition().
            If this is None, the algorithm starts a fresh sequence of releases. (default: None)
        last_partition : list of lists of int
            Output from the last call to partition(). If this is None, the algorithm
            starts a fresh sequence of releases. (default: None)
        '''
        self._m = m
        self._last_df = last_df
        self._last_output = last_partition

    def partition(self, df):
        '''
        Partitions the given data frame into equivalence classes satisfying m-invariance.

        Parameters
        ----------
        df : pandas.DataFrame
            Raw data frame which will be partitioned. Must contain a column ID which uniquely
            identifies every record.

        Returns
        -------
        A tuple. The first element is the partitioning of df which can be passed to a generalisation
        schema to create a generalised/anonymised version of df. The second element is a dict
        mapping the index of an equivalence class within the partition to a dictionary mapping
        sensitive values to the number of counterfeits with this sensitive value in the respective
        equivalence class.
        '''
        existing, insertions = self._preprocess(df)

        buckets = self._divide(existing)
        buckets, insertions, counterfeits = self._balance(buckets, insertions)
        buckets, more_counterfeits = self._assign(buckets, insertions)
        result = self._split(buckets)

        counterfeits += more_counterfeits

        counterfeit_statistics = self._create_counterfeit_statistics(counterfeits)

        self._last_df = df
        self._last_output = result
        return result, counterfeit_statistics

    def _preprocess(self, df):
        existing = None # TODO: records which were already added in the past
        insertions = None # TODO: records which have been added in this release

        return existing, insertions

    def _divide(self, existing):
        buckets = None # TODO: divide the records which exist in previous releases into buckets

        return buckets

    def _balance(self, buckets, insertions):
        buckets = buckets # TODO: balance the buckets
        remaining_insertions = insertions # TODO: insertions without the records used during balancing
        counterfeits = None # TODO: information about counterfeits used to balance the buckets

        return buckets, remaining_insertions, counterfeits

    def _assign(self, buckets, remaining_insertions):
        buckets = buckets # TODO: buckets after assigning the remaining insertions
        counterfeits = None # TODO: information about the counterfeits added to keep the buckets balanced (the paper assumes that the insertions are m-eligible, but if this is not the case, you can just add further counterfeits)

        return buckets, counterfeits

    def _split(self, buckets):
        partitioning = [] # TODO: see e.g. Mondrian: list of lists containing the indices of records from the data frame (df) which form an equivalence class

        return partitioning

    def _create_counterfeit_statistics(self, counterfeits):
        counterfeit_statistics = None # TODO: see return value of partition()

        return counterfeit_statistics
