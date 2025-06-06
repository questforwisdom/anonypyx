from anonypyx.dlx.multiset_dlx import *

import pytest

def test_all_part_of_only_solution():
    target = [1, 2, 3]
    sparse_rows = [
        [0, 2],
        [1, 2],
        [1],
        [2]
    ]

    problem = ExactMultisetCover(target, sparse_rows)
    marked = problem.part_of_any_solution()

    assert marked == {0, 1, 2, 3}

def test_only_one_solution():
    target = [1, 1, 2]
    sparse_rows = [
        [0],
        [0, 2],
        [1, 2],
        [0, 1]
    ]

    problem = ExactMultisetCover(target, sparse_rows)
    marked = problem.part_of_any_solution()

    assert marked == {1, 2}

def test_no_solution():
    target = [1, 2, 1, 2]
    sparse_rows = [
        [0, 1, 3],
        [0, 2, 3],
        [1, 2],
        [2, 3]
    ]

    problem = ExactMultisetCover(target, sparse_rows)
    marked = problem.part_of_any_solution()

    assert marked == set()

def test_unsorted_sparse_rows():
    target = [1, 2, 1]
    sparse_rows = [
        [2, 1],
        [2, 0, 1],
        [1, 0],
        [0]
    ]

    problem = ExactMultisetCover(target, sparse_rows)
    marked = problem.part_of_any_solution()

    assert marked == {0, 2}

def test_only_one_column():
    target = [2]
    sparse_rows = [
        [0],
        [0],
        [0],
    ]

    problem = ExactMultisetCover(target, sparse_rows)
    marked = problem.part_of_any_solution()

    assert marked == {0, 1, 2}

def test_multiple_solutions():
    target = [2, 1, 1, 2]
    sparse_rows = [
        [0, 3],
        [0, 1],
        [2, 3],
        [1, 2],
        [0, 1, 2, 3]
    ]
    # solutions are {0, 4} and {0, 1, 2}

    problem = ExactMultisetCover(target, sparse_rows)
    marked = problem.part_of_any_solution()

    assert marked == {0, 1, 2, 4}
