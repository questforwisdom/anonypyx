import json

import anonypyx.generalisation

def save_schema(schema, filename):
    '''
    Saves a generalisation schema to the disk.

    Parameters
    ----------

    schema : GeneralisationSchema
        The schema object to serialise.
    filename : str
        Path to the file in which the generalisation schema will be stored.
    '''
    json_dict = schema.to_json_dict()

    if isinstance(schema, anonypyx.generalisation.HumanReadable):
        json_dict['schema_type'] = 'HumanReadable'
    elif isinstance(schema, anonypyx.generalisation.MachineReadable):
        json_dict['schema_type'] = 'MachineReadable'
    elif isinstance(schema, anonypyx.generalisation.Microaggregation):
        json_dict['schema_type'] = 'Microaggregation'
    elif isinstance(schema, anonypyx.generalisation.RawData):
        json_dict['schema_type'] = 'RawData'

    content = json.dumps(json_dict)

    with open(filename, 'w') as f:
        f.write(content)

def load_schema(filename):
    '''
    Loads a generalisation schema from the disk.

    Parameters
    ----------

    filename : str
        Path to the file which describes the generalisation schema to load.

    Returns
    -------

    A GeneralisationSchema instance as specified by the file.
    '''
    content = None

    with open(filename) as f:
        content = f.read()

    json_dict = json.loads(content)

    schema_type = json_dict['schema_type']

    if schema_type == 'HumanReadable':
        return anonypyx.generalisation.HumanReadable.from_json_dict(json_dict)
    elif schema_type == 'MachineReadable':
        return anonypyx.generalisation.MachineReadable.from_json_dict(json_dict)
    elif schema_type == 'Microaggregation':
        return anonypyx.generalisation.Microaggregation.from_json_dict(json_dict)
    elif schema_type == 'RawData':
        return anonypyx.generalisation.RawData.from_json_dict(json_dict)
    else:
        raise NotImplementedError(f'Unsupported schema type "{schema_type}"')
