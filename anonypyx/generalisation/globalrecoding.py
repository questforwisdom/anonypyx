from anonypyx.generalisation.schema import GeneralisedSchema, build_column_groups

import pandas as pd

class GlobalRecoding(GeneralisedSchema):
    '''
    Generalised schema for global recoding according to a predefined
    generalisation taxonomy. The taxonomy defines a hierarchy of valid
    generalisations for raw values. A global recoding replaces all
    occurrences of a raw value with the same generalised value.
    This class also ensures that all descendants of a generalised value
    (with respect to the taxonomy) are also replaced by it. For instance,
    if the taxonomy states that the attribute age is generalised to intervals
    of 10 years ([0, 9], [10, 19], ...) and if the value 22 is replaced by [20, 29],
    then 20, 21, 23, ..., 29 are also replaced by [20, 29] to avoid ambiguity.
    '''

    @classmethod
    def from_json_dict(cls, json_dict):
        taxonomies = {}

        for qi_column, definition in json_dict['taxonomy'].items():
            node_map = {value: Taxonomy(value) for value in definition['generalised']}
            raw_value_map = {}

            for value, specialisations in definition['definitions'].items():
                for specialisation in specialisations:
                    if specialisation in definition['generalised']:
                        node_map[value].add_generalised(node_map[specialisation])
                    else:
                        if not value in raw_value_map:
                            raw_value_map[value] = []
                        raw_value_map[value].append(specialisation)

            for value, raw_specialisations in raw_value_map.items():
                node_map[value].add_raw_values(raw_specialisations)

            root = None
            for node in node_map.values():
                if node.parent() is None:
                    root = node

            taxonomies[qi_column] = root

        return GlobalRecoding(taxonomies, json_dict['unaltered'])

    def __init__(self, taxonomies, unaltered):
        '''
        Constructor.

        Parameters
        ----------
        taxonomies : dict mapping str to anonypyx.generalisation.globalrecoding.Taxonomy
            Dictionary mapping the column names of quasi-identifiers to the generalisation
            Taxonomy according to which they are generalised.
        unaltered : list of str
            List of column names which are not quasi-identifiers.
        '''
        super().__init__(unaltered)
        self._qi_taxonomies = taxonomies

    def to_json_dict(self):
        result = {'taxonomy': {}}

        for qi_column, taxonomy in self._qi_taxonomies.items():
            definition = {'generalised': [], 'definitions': {}}

            def tree_walk(root):
                values = []
                definition['generalised'].append(root.value())

                for child in root.children():
                    if child.is_raw_value():
                        values += sorted(list(child.raw_values()))
                    else:
                        values.append(child.value())

                definition['definitions'][root.value()] = values

                for child in root.children():
                    if not child.is_raw_value():
                        tree_walk(child)

            tree_walk(taxonomy)
            result['taxonomy'][qi_column] = definition

        result['unaltered'] = self._unaltered

        return result

    def match(self, df, record, on):
        recoded_record = self._recode_record(record, df, on)
        return self._recoded_query(df, recoded_record)

    def intersect(self, record_a, record_b, on, take_left, take_right):
        result = {}
        for column in on:
            value = self._intersect_values(column, record_a[column], record_b[column])

            if value is None:
                return None

            result[column] = value

        for column in take_left:
            result[column] = record_a[column]

        for column in take_right:
            result[column] = record_b[column]

        return result

    def values_for(self, record, column):
        if column not in self._qi_taxonomies:
            return record[column]

        taxonomy_node = self._qi_taxonomies[column].find_value(record[column])

        if taxonomy_node.is_raw_value():
            return {record[column]}

        return taxonomy_node.raw_values()

    def quasi_identifier(self):
        return list(self._qi_taxonomies.keys())

    def set_cardinality(self, record, on):
        cardinality = 1

        for column in on:
            if column in self._qi_taxonomies:
                taxonomy_node = self._qi_taxonomies[column].find_value(record[column])

                if not taxonomy_node.is_raw_value():
                    cardinality *= taxonomy_node.cardinality(None)

        return cardinality

    def select(self, df, query):
        recoded_query = self._recode_query(query, df)

        if recoded_query is None:
            # some restriction in query is disjunct to df
            return []
        return self._recoded_query(df, recoded_query).index

    def query_overlap(self, record, query):
        cardinality = 1

        for column, restriction in query.items():
            if column in self._qi_taxonomies:
                if not isinstance(restriction, set):
                    restriction = set(range(restriction[0], restriction[1] + 1))

                taxonomy_node = self._qi_taxonomies[column].find_value(record[column])

                matches = taxonomy_node.cardinality(restriction)

                if matches == 0:
                    return 0

                cardinality *= matches if not taxonomy_node.is_raw_value() else 1
            else:
                if isinstance(restriction, set):
                    if record[column] not in restriction:
                        return 0
                else:
                    if record[column] < restriction[0] or record[column] > restriction[1]:
                        return 0

        return cardinality

    def _generalise_quasi_identifiers(self, df, partitions):
        # added for the sake of completeness
        # rather time consuming algorithm
        # global recoding should be used with dedicated algorithms if possible
        columns = self.quasi_identifier()
        data = {col: [] for col in columns}
        for column in columns:
            replacements = []

            for i, partition in enumerate(partitions):
                value_this = self._generalisation_of(column, df.loc[partition][column].unique())
                node_this = self._qi_taxonomies[column].find_value(value_this)

                # invariant: there is at most one distinct value on the same root-leaf-path as value_this
                for j, value_other in enumerate(replacements):
                    node_other = self._qi_taxonomies[column].find_value(value_other)

                    if on_same_path(node_this, node_other):
                        if node_this.level() <= node_other.level():
                            replacements[j] = value_this
                        else:
                            value_this = value_other
                            break

                replacements.append(value_this)

            data[column] = replacements
            data['group_id'] = range(len(partitions))

        columns.append('group_id')

        return pd.DataFrame(data, columns=columns)

    def _intersect_values(self, column, value_a, value_b):
        if column in self._qi_taxonomies.keys():
            node_a = self._qi_taxonomies[column].find_value(value_a)
            node_b = self._qi_taxonomies[column].find_value(value_b)

            if not on_same_path(node_a, node_b):
                return None

            if node_a.level() <= node_b.level():
                return value_b
            else:
                return value_a
        else:
            if value_a == value_b:
                return value_a

    def _recode_record(self, record, df, on):
        recoded_record = {}

        for column in on:
            if column in self._qi_taxonomies:
                existing_values = list(df[column].unique())
                recoded_record[column] = []
                node_record = self._qi_taxonomies[column].find_value(record[column])
                for value in existing_values:
                    node_value = self._qi_taxonomies[column].find_value(value)

                    if on_same_path(node_value, node_record):
                        recoded_record[column].append(value)
            else:
                recoded_record[column] = [record[column]]

        return recoded_record

    def _generalisation_of(self, column, values):
        if len(values) == 1:
            for value in values:
                return value

        nodes_by_level = {}
        remaining = 0

        for value in values:
            node = self._qi_taxonomies[column].find_value(value)

            if node.level() not in nodes_by_level:
                nodes_by_level[node.level()] = []

            if not node in nodes_by_level[node.level()]:
                nodes_by_level[node.level()].append(node)
                remaining += 1

        max_level = max(nodes_by_level.keys())

        for level in range(max_level + 1):
            if level not in nodes_by_level:
                nodes_by_level[level] = []

        for level in range(max_level, 0, -1):
            if remaining == 1:
                break

            for node in nodes_by_level[level]:
                parent = node.parent()


                if not parent in nodes_by_level[parent.level()]:
                    nodes_by_level[parent.level()].append(parent)
                else:
                    remaining -= 1

        for level in range(0, max_level + 1):
            if len(nodes_by_level[level]) > 0:
                node_generalised = nodes_by_level[level][0]

                if node_generalised.is_raw_value():
                    # the case that no generalisation is needed is already covered above
                    # this case here happens if there are multiple distinct raw values AND
                    # all of them belong to the same TaxonomyLeaves instance
                    return node_generalised.parent().value()
                return node_generalised.value()

    def _recode_query(self, query, df):
        recoded_query = {}

        for column, restriction in query.items():
            if not isinstance(restriction, set):
                 restriction = set(range(restriction[0], restriction[1] + 1))

            if column in self._qi_taxonomies:
                recoded_query[column] = []
                existing_values = list(df[column].unique())


                for value in existing_values:
                    overlap = self.query_overlap({column: value}, {column: restriction}) 

                    if overlap > 0:
                        recoded_query[column].append(value)

                if len(recoded_query[column]) == 0:
                    # query does not match any value in df
                    return None
            else:
                recoded_query[column] = list(restriction)

        return recoded_query

    def _recoded_query(self, df, recoded_query):
        query = []

        for column in recoded_query.keys():
            subquery = []

            for value in recoded_query[column]:
                if isinstance(value, str):
                    subquery.append(f'`{column}` == "{value}"')
                else:
                    subquery.append(f'`{column}` == {value}')

            query.append('(' + ' or '.join(subquery) + ')')

        query = ' and '.join(query)

        return df.query(query)

class Taxonomy:
    '''
    Generalisation taxonomy for a single attribute.
    The taxnonomy is a hierarchy (tree) where leaves are raw values
    and inner nodes represent generalised values such that the value of 
    an inner node is a generalised representation of all nodes contained
    in the subtree rooted in this inner node.
    '''
    def __init__(self, value):
        '''
        Constructor.

        Parameters
        ----------
        value : str
            Generalised value which is used to represent the subtree rooted in this node.
        '''
        self._value = value
        self._children = []
        self._raw_value_iterators = []
        self._parent = None
        self._level = 0

    def add_generalised(self, child_node):
        '''
        Adds a new inner node (i.e. a generalised value) as a child of this node.

        Parameters
        ----------
        child_node : anonypyx.generalisation.globalrecoding.Taxonomy
            The inner node to add as a child to this node.    

        Returns
        -------
        Reference to this object (allows to "chain" calls when building the tree manually).
        '''
        self._children.append(child_node)
        child_node.set_parent(self)
        child_node.update_level(self._level + 1)
        return self

    def add_raw_values(self, raw_values):
        '''
        Adds leaves (i.e. raw values) as children of this node.

        Parameters
        ----------
        raw_values : iterable
            The raw values of which are directly generalised by this node.

        Returns
        -------
        Reference to this object (allows to "chain" calls when building the tree manually).
        '''
        self.add_generalised(TaxonomyLeaves(raw_values))
        return self

    def find_value(self, value):
        if self._value == value:
            return self

        for child in self._children:
            node = child.find_value(value)
            if node is not None:
                return node

        return None

    def is_raw_value(self):
        return False

    def raw_values(self):
        result = set()

        for child in self._children:
            result = result.union(child.raw_values())

        return result

    def set_parent(self, parent_node):
        self._parent = parent_node

    def parent(self):
        return self._parent

    def children(self):
        return self._children

    def level(self):
        return self._level

    def update_level(self, level):
        self._level = level
        for child in self._children:
            child.update_level(level + 1)

    def value(self):
        return self._value

    def cardinality(self, restriction):
        return sum(child.cardinality(restriction) for child in self._children)

class TaxonomyLeaves:
    def __init__(self, raw_values):
        self._raw_values = raw_values
        self._parent = None
        self._level = 0

    def find_value(self, value):
        if value in self._raw_values:
            return self
        return None

    def is_raw_value(self):
        return True

    def raw_values(self):
        return set(self._raw_values)

    def set_parent(self, parent_node):
        self._parent = parent_node

    def parent(self):
        return self._parent

    def update_level(self, level):
        self._level = level

    def level(self):
        return self._level

    def cardinality(self, restriction):
        if restriction is None:
            return len(list(self._raw_values))
        else:
            return len(list(val in self._raw_values for val in self._raw_values if val in restriction))

def on_same_path(node_1, node_2):
    if node_1 == node_2:
        return True

    if node_1.level() == node_2.level():
        return False

    top = node_1 if node_1.level() < node_2.level() else node_2
    bottom = node_2 if node_1.level() < node_2.level() else node_1

    cursor = bottom

    while cursor.parent() is not None:
        cursor = cursor.parent()

        if cursor == top:
            return True

    return False
