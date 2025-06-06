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

    def finalise(self):
        '''
        Call this method once all releases have been observed to complete the attack.
        '''
        pass

def parse_prior_knowledge(prior_knowledge, id_callback):
    '''
    Helper function for attackers which parses the prior knowledge.

    Parameters
    ----------
    prior_knowledge : pandas.DataFrame
        A data frame containing the attacker's prior knowledge about their targets. It must use the
        same generalisation schema as the data frames the attacker will observe() (minus the column
        'count'). Furthermore, it must contain a column 'ID' which uniquely identifies every target.
        The ID must start at zero and increase strictly monotonically (i.e. IDs must be 0, 1, ...,
        num_targets - 1). The data frame may contain multiple rows with the same ID which is
        interpreted as an attacker having alternative hypotheses about the target. At least one
        hypothesis (row) for every ID must be true.
    id_callback : function pointer with parameters int and pandas.DataFrame
        This function is called for every ID (first parameter) with a data frame containing all rows
        for this ID. The ID column is not included in the data frame.

    Returns
    -------
    The number of targets described by the prior knowledge.
    '''
    num_targets = prior_knowledge['ID'].max() + 1
    for target_id in range(num_targets):
        matcher = prior_knowledge['ID'] == target_id
        target_knowledge = prior_knowledge[matcher].drop('ID', axis=1)
        id_callback(target_id, target_knowledge)
    return num_targets

