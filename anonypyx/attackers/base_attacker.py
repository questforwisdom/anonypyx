class BaseAttacker:
    def observe(self, release, present_columns, present_targets):
        '''
        Observe a new data release.

        Parameters
        ----------
        release : pandas.DataFrame
            The generalised data frame of the release.
        present_columns : list of str
            Column names (before generalisation) which are present in the release.
        present_targets : list of int
            Identifiers of targeted individuals which are present in the release.
        '''
        raise NotImplementedError()

    def predict(self, target_id, column):
        '''
        Predicts the value for a target.

        Parameters
        ----------
        target_id : int
            Identifier of the target for which the value is predicted.
        column : str
            Column name (before generalisation) for which the value is predicted.A

        Returns
        -------
        A dictionary mapping the predicted values to weights. A larger weight indicates
        a higher confidence in the predicted value.
        '''
        raise NotImplementedError()
