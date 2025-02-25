from collections import Counter

from pandas import testing as tm

def assert_data_set_equal(left_df, right_df):
    # pandas is unable to ignore the row order when comparing dataframes for equality
    # (there is a parameter for ignoring the column order though)
    # this is a workaround

    # ensure that the columns are equal (ignoring order)
    left_columns = left_df.columns.to_list()
    right_columns = right_df.columns.to_list()

    assert set(left_columns) == set(right_columns)

    # ensure that rows are ordered similarily
    left_sorted = left_df.sort_values(left_columns)
    right_sorted = right_df.sort_values(left_columns)

    # reset the index so that pandas forgets the old order
    left_sorted = left_sorted.copy().reset_index(drop=True)
    right_sorted = right_sorted.copy().reset_index(drop=True)

    # do the actual comparison, ignoring column order
    tm.assert_frame_equal(left_sorted, right_sorted, check_like=True)

def assert_multiset_equal(left_list, right_list):
    # equality check for lists where order of elements is ignored

    left_counter = Counter([tuple(element) for element in left_list])
    right_counter = Counter([tuple(element) for element in right_list])

    assert left_counter == right_counter

