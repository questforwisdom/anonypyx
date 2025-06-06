import numpy as np

import anonypyx.dlx
from anonypyx.attackers.util import split_columns
from anonypyx.attackers.base_attacker import BaseAttacker, parse_prior_knowledge

class Trajectory:
    def __init__(self, trajectory, record, permutations):
        self._trajectory = trajectory
        self._record = record
        self._permutations = permutations

    def extend(self, release, schema, shared_columns, take_left, take_right, trajectory_offset):
        new_candidates = []
        matches = schema.match(release, self._record, on=shared_columns)

        for match in matches.index:
            new_trajectory = self._trajectory + [match + trajectory_offset]
            new_record = schema.intersect(self._record, matches.loc[match], on=shared_columns, take_left=take_left, take_right=take_right)
            new_permutations = self._permutations * matches.loc[match]['count']
            new_candidates.append(Trajectory(new_trajectory, new_record, new_permutations))

        return new_candidates

    def mark_as_absent(self, trajectory_offset):
        return Trajectory(self._trajectory + [trajectory_offset], self._record, self._permutations)

    def predict(self, column, schema):
        return schema.values_for(self._record, column)

    def equivalent_permutations(self):
        return self._permutations

    def to_matrix_row(self, row_length):
        return self._trajectory

class TrajectoryAttacker(BaseAttacker):
    def __init__(self, prior_knowledge, present_columns, schema):
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
        present_columns : list of str
            The names of the columns in the original data frames (before generalisation) which are
            contained in prior_knowledge (i.e. those for which the attacker knows the exact values).
        schema : anonypyx.generalisation.GeneralisedSchema
            The generalisation schema used by the data frames the attacker will observe() and by
            prior_knowledge.
        '''
        self._record_counts = []
        self._target_trajectories = []
        self._target_known_columns = []
        self._schema = schema
        self._prepare_candidate_set(prior_knowledge, present_columns)

    def _prepare_candidate_set(self, prior_knowledge, present_columns):
        def id_callback(target_id, target_knowledge):
            trajectories = []
            for _, row in target_knowledge.iterrows():
                trajectories.append(Trajectory([target_id], row, 1))
            self._target_trajectories.append(trajectories)
            self._target_known_columns.append(present_columns[:])

        num_targets = parse_prior_knowledge(prior_knowledge, id_callback)

        self._record_counts = [1] * num_targets

    def observe(self, release, present_columns, present_targets):
        start_present = len(self._record_counts)
        start_absent = start_present

        num_absent = len(self._target_trajectories) - len(present_targets)

        if num_absent > 0:
            start_present += 1

        for target_id, trajectories in enumerate(self._target_trajectories):
            if target_id in present_targets:
                take_left, shared_columns, take_right = split_columns(self._target_known_columns[target_id], present_columns)

                new_trajectories = []
                for trajectory in trajectories:
                    new_trajectories += trajectory.extend(release, self._schema, shared_columns, take_left, take_right, start_present)

                self._target_trajectories[target_id] = new_trajectories
                self._target_known_columns[target_id] += take_right
            else:
                self._target_trajectories[target_id] = [t.mark_as_absent(start_absent) for t in trajectories]

        if num_absent > 0:
            self._record_counts += [num_absent]
        self._record_counts += release['count'].to_list()

    def predict(self, target_id, column):
        if column not in self._target_known_columns[target_id]:
            return None

        # TODO: temporary or maybe forever? do not weight the predictions
        result = {}
        for trajectory in self._target_trajectories[target_id]:
            value_set = trajectory.predict(column, self._schema)
            for value in value_set:
                result[value] = result.get(value, 1)
                # old_count = result.get(value, 0)
                # result[value] = old_count + trajectory.equivalent_permutations()

        return result

    def finalise(self):
        target = self._record_counts
        matrix = []

        for trajectories in self._target_trajectories:
            for trajectory in trajectories:
                matrix.append(trajectory.to_matrix_row(len(target)))

        problem = anonypyx.dlx.ExactMultisetCover(target, matrix)
        consistent_rows = problem.part_of_any_solution()

        row_index = 0

        consistent_trajectories = []

        for trajectories in self._target_trajectories:
            for_this_target = []
            for trajectory in trajectories:
                if row_index in consistent_rows:
                    for_this_target.append(trajectory)
                row_index += 1
            consistent_trajectories.append(for_this_target)
        self._target_trajectories = consistent_trajectories
