import numpy as np
import exact_multiset_cover as ec

from anonypyx.attackers.util import split_columns
from anonypyx.attackers.base_attacker import BaseAttacker

class Trajectory:
    def __init__(self, trajectory, record, permutations):
        self._trajectory = trajectory
        self._record = record
        self._permutations = permutations

    def extend(self, release, schema, shared_columns, take_left, take_right, trajectory_offset):
        new_candidates = []
        matches = schema.match(release, self._record, on=shared_columns)

        for match in matches:
            new_trajectory = self._trajectory + [match + trajectory_offset]
            new_record = schema.intersect(self._record, release.loc[match], on=shared_columns, take_left=take_left, take_right=take_right)
            new_permutations = self._permutations * release.loc[match]['count']
            new_candidates.append(Trajectory(new_trajectory, new_record, new_permutations))

        return new_candidates

    def mark_as_absent(self, trajectory_offset):
        return Trajectory(self._trajectory + [trajectory_offset], self._record, self._permutations)

    def predict(self, column, schema):
        return schema.values_for(self._record, column)

    def equivalent_permutations(self):
        return self._permutations

    def to_matrix_row(self, row_length):
        row = np.zeros(row_length)
        for row_id in self._trajectory:
            row[row_id] = 1
        return row

class TrajectoryAttacker(BaseAttacker):
    def __init__(self, prior_knowledge, present_columns, schema):
        self._record_counts = []
        self._target_trajectories = []
        self._target_known_columns = []
        self._schema = schema
        self._prepare_candidate_set(prior_knowledge, present_columns)

    def _prepare_candidate_set(self, prior_knowledge, present_columns):

        num_targets = prior_knowledge.iloc[:, 0].max() + 1
        for target_id in range(num_targets):
            matcher = prior_knowledge.iloc[:, 0] == target_id
            target_knowledge = prior_knowledge[matcher].iloc[:, 1:]
            trajectories = []
            for _, row in target_knowledge.iterrows():
                trajectories.append(Trajectory([target_id], row, 1))
            self._target_trajectories.append(trajectories)
            self._target_known_columns.append(present_columns[:])

        self._record_counts = [1] * num_targets

    def observe(self, release, present_columns, present_targets):
        trajectory_offset = len(self._record_counts)

        num_absent = len(self._target_trajectories) - len(present_targets)

        for target_id, trajectories in enumerate(self._target_trajectories):
            if target_id in present_targets:
                take_left, shared_columns, take_right = split_columns(self._target_known_columns[target_id], present_columns)

                new_trajectories = []
                for trajectory in trajectories:
                    new_trajectories += trajectory.extend(release, self._schema, shared_columns, take_left, take_right, trajectory_offset + 1)

                self._target_trajectories[target_id] = new_trajectories
                self._target_known_columns[target_id] += take_right
            else:
                self._target_trajectories[target_id] = [t.mark_as_absent(trajectory_offset) for t in trajectories]

        self._record_counts += [num_absent] + release['count'].to_list()

    def predict(self, target_id, column):
        if column not in self._target_known_columns[target_id]:
            return None

        result = {}
        for trajectory in self._target_trajectories[target_id]:
            value_set = trajectory.predict(column, self._schema)
            for value in value_set:
                old_count = result.get(value, 0)
                result[value] = old_count + trajectory.equivalent_permutations()

        return result

    def _remove_zero_columns(self, target, matrix):
        to_delete = []
        for i, value in enumerate(target):
            if value == 0:
                to_delete.append(i)

        target = np.delete(target, to_delete)
        matrix = np.delete(matrix, to_delete, axis=1)

        return target, matrix

    def _consistent_rows(self, exact_cover_solutions):
        consistent_rows = set()
        for solution in exact_cover_solutions:
            for row_index in solution:
                consistent_rows.add(row_index)
        return consistent_rows

    def prune_multiset_exact_cover(self):
        target = np.array(self._record_counts, dtype=ec.io.DTYPE_FOR_ARRAY)
        matrix = []

        for trajectories in self._target_trajectories:
            for trajectory in trajectories:
                matrix.append(trajectory.to_matrix_row(len(target)))

        matrix = np.array(matrix, dtype=ec.io.DTYPE_FOR_ARRAY)

        target, matrix = self._remove_zero_columns(target, matrix)

        solutions = ec.get_all_solutions(matrix, target=target)

        consistent_rows = self._consistent_rows(solutions)

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
