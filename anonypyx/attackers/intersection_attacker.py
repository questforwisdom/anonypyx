from anonypyx.attackers.util import split_columns
from anonypyx.attackers.base_attacker import BaseAttacker, parse_prior_knowledge

class SensitiveValueSet:
    def __init__(self, quasi_identifier_knowledge, quasi_identifiers, sensitive_column, schema):
        self._knowledge = quasi_identifier_knowledge
        self._quasi_identifier = quasi_identifiers
        self._sensitive_column = sensitive_column
        self._value_set = None
        self._schema = schema

    def update(self, release):
        matches = self._schema.match(release, self._knowledge, on=self._quasi_identifier)

        value_set = set(matches[self._sensitive_column].unique())

        if self._value_set is None:
            self._value_set = value_set
        else:
            self._value_set = self._value_set.intersection(value_set)

    def values_for(self, column):
        if column == self._sensitive_column:
            return {v: 1 for v in self._value_set}

        return {v: 1 for v in self._schema.values_for(self._quasi_identifier, column)}

class IntersectionAttacker(BaseAttacker):
    def __init__(self, prior_knowledge, quasi_identifiers, sensitive_column, schema):
        '''
        Constructor.

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
        quasi_identifiers : list of str
            The names of the columns in the original data frames (before generalisation) which act as
            quasi-identifiers (i.e. those for which the attacker knows the exact values).
        sensitive_column : str
            The names of the column in the original data frames (before generalisation) which acts as
            the sensitive attribute (i.e. the one the attacker attempts to reconstruct).
        schema : anonypyx.generalisation.GeneralisedSchema
            The generalisation schema used by the data frames the attacker will observe() and by
            prior_knowledge.
        '''
        self._candidates = []

        def id_callback(target_id, target_knowledge):
            for _, row in target_knowledge.iterrows():
                # TODO: only one row per target supported right now
                self._candidates.append(SensitiveValueSet(row, quasi_identifiers, sensitive_column, schema))

        # TODO: use column name 'ID' instead of fixed position
        num_targets = parse_prior_knowledge(prior_knowledge, id_callback)

    def observe(self, release, present_columns, present_targets):
        for target_id in present_targets:
            self._candidates[target_id].update(release)

    def predict(self, target_id, column):
        return self._candidates[target_id].values_for(column)
