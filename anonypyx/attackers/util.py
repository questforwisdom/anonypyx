def split_columns(values_known, values_released):
    left = []
    right = []
    both = []
    for col in values_released:
        if col in values_known:
            both.append(col)
        else:
            right.append(col)
    for col in values_known:
        if col not in values_released:
            left.append(col)

    return left, both, right
