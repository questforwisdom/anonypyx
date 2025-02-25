from anonypyx.attackers.util import split_columns
from anonypyx.attackers.base_attacker import BaseAttacker

class SensitiveValueSet:
    def __init__(self, quasi_identifier_knowledge, quasi_identifiers, sensitive_column, schema):
        self._knowledge = quasi_identifier_knowledge
        self._quasi_identifier = quasi_identifiers
        self._sensitive_column = sensitive_column
        self._value_set = None
        self._schema = schema

    def update(self, release):
        matches = self._schema.match(release, self._knowledge, on=self._quasi_identifier)

        value_set = set(release.loc[matches][self._sensitive_column].unique())

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
        self._candidates = []
        num_targets = prior_knowledge.iloc[:, 0].max() + 1
        for target_id in range(num_targets):
            matcher = prior_knowledge.iloc[:, 0] == target_id
            target_knowledge = prior_knowledge[matcher].iloc[:, 1:]
            for _, row in target_knowledge.iterrows():
                # TODO: only one row per target supported right now
                self._candidates.append(SensitiveValueSet(row, quasi_identifiers, sensitive_column, schema))

    def observe(self, release, present_columns, present_targets):
        for target_id in present_targets:
            self._candidates[target_id].update(release)

    def predict(self, target_id, column):
        return self._candidates[target_id].values_for(column)
