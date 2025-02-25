def reconstruct_records(candidates, generalised_columns):
    record_mapping = {}
    reconstruction_possible = set()

    for entry in candidates:
        trajectory = entry[0]
        candidate = entry[1]
        identifier = trajectory[0]

        existing_candidate = record_mapping.get(identifier, {})
        if identifier not in record_mapping:
            record_mapping[identifier] = candidate
            reconstruction_possible.add(identifier)
        else:
            old_candidate = record_mapping[identifier]
            
            if old_candidate != candidate:
                reconstruction_possible.remove(identifier)

    records = {}
    for identifier in reconstruction_possible:
        candidate = record_mapping[identifier]
        record = generalised_columns.collapse_record(candidate)
        if record is not None:
            records[identifier] = record

    return records
