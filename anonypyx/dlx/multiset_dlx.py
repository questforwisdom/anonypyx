# replacement for exact_multiset_cover
# exact_multiset_cover is designed for small solutions, not for large data sets
# problems include
# - the entire (sparse) matrix is constructed in memory
# - all solutions are identified and stored in the same buffer (already more than 4GB for 100 data points and 5 releases)
# - the problem is solved twice: once to determine the number of solutions to create the buffer and once to store the solutions
#
# this file just provides the functionality required for the trajectory attacker with optimisations
# maybe we can turn it into some C extension again in the future for further speed up...

# TODO: integrate into trajectory attacker
# TODO: remove dependency to exact_multiset_cover

class Node:
    def __init__(self, counter, multiplcity, header):
        self.counter = counter
        self.multiplicity = multiplcity
        self.header = header
        self.left = self
        self.right = self
        self.up = self
        self.down = self
        self.dangling_next = self
        self.dangling_previous = self

    def insert_horizontally(self, other):
        other.right = self
        other.left = self.left
        self.left.right = other
        self.left = other
        return other

    def insert_horizontally_after(self, other):
        start = self.right
        return start.insert_horizontally(other)

    def insert_vertically(self, other):
        other.down = self
        other.up = self.up
        self.up.down = other
        self.up = other
        return other

    def insert_vertically_after(self, other):
        start = self.down
        return start.insert_vertically(other)

    def insert_dangling_after(self, other):
        start = self.dangling_next
        other.dangling_next = start
        other.dangling_previous = start.dangling_previous
        start.dangling_previous.dangling_next = other
        start.dangling_previous = other
        return other

    def new_dangling_stack(self):
        self.dangling_next = self
        self.dangling_previous = self
        return self

    def cover_horizontally(self):
        if self.right == self:
            return None
        self.right.left = self.left
        self.left.right = self.right
        return self.right

    def uncover_horizontally(self):
        self.right.left = self
        self.left.right = self

    def uncover_vertically(self):
        self.down.up = self
        self.up.down = self

    def cover_vertically(self):
        if self.down == self:
            return None
        self.down.up = self.up
        self.up.down = self.down
        return self.down

    def cover_row(self):
        next_node = self.right
        while next_node != self:
            col = next_node.header
            next_node.cover_vertically()
            col.counter -= 1
            next_node = next_node.right

    def uncover_row(self):
        next_node = self.left
        while next_node != self:
            col = next_node.header
            col.counter += 1
            next_node.uncover_vertically()
            next_node = next_node.left

    def cover_column(self, col):
        col.multiplicity -= self.multiplicity
        if col.multiplicity == 0:
            col.cover_horizontally()
            next_row = col.down
            while next_row != col:
                if next_row != self: # selected row is removed through cover_column calls for the other columns
                    next_row.cover_row()
                next_row = next_row.down
        else:
            self.cover_vertically()
            col.counter -= 1
            dangling_stack = self.new_dangling_stack()
            next_row = col.down
            while next_row != col:
                if next_row.multiplicity > col.multiplicity and next_row != self:
                    next_row.cover_vertically()
                    next_row.cover_row()
                    col.counter -= 1
                    dangling_stack = dangling_stack.insert_dangling_after(next_row)
                next_row = next_row.down

    def uncover_column(self, col):
        if col.multiplicity == 0:
            next_row = col.up
            while next_row != col:
                if next_row != self:
                    next_row.uncover_row()
                next_row = next_row.up
            col.uncover_horizontally()
        else:
            next_row = self.dangling_previous
            while next_row != self:
                next_row.uncover_row()
                next_row.uncover_vertically()
                col.counter += 1
                next_row = next_row.dangling_previous
            col.counter += 1
            self.uncover_vertically()
        col.multiplicity += self.multiplicity

    def empty_horizontal_list(self):
        return self.right == self

    def at_horizontal(self, index):
        i = 0
        col = self.right
        while col != self:
            if i == index:
                return col
            i += 1
            col = col.right
        raise ValueError(f'Index {i} is out of bounds.')

class ExactMultisetCover:
    def __init__(self, target, sparse_rows):
        self._sparse_matrix = self._create_headers(target)
        self._populate(sparse_rows)
        self._marked = set()

    def part_of_any_solution(self):
        solution = []
        self._explore(solution)
        return self._marked

    def _create_headers(self, target):
        sparse_matrix = Node(0, 0, None)
        column_header = sparse_matrix

        for col_num in range(len(target)):
            multiplicity = target[col_num]
            if multiplicity <= 0:
                raise ValueError('Target multiplicity must be greater than zero.')
            new_node = Node(0, multiplicity, None)
            column_header = column_header.insert_horizontally_after(new_node)
        return sparse_matrix

    def _populate(self, sparse_rows):
        # sparse rows: list of lists of indicies of rows which are filled with a 1, all others are 0
        for row_num in range(len(sparse_rows)):
            sparse_row = sparse_rows[row_num]
            row_list = None
            for col_num in sparse_row:
                col = self._sparse_matrix.at_horizontal(col_num)
                node = Node(row_num, 1, col)
                row_list = row_list.insert_horizontally_after(node) if row_list is not None else node
                col.up.insert_vertically_after(node)
                col.counter += 1

    def _explore(self, solution, last_col=None, last_row_num=0):
        if self._sparse_matrix.empty_horizontal_list():
            for row_num in solution:
                self._marked.add(row_num)
            return

        col = self._choose_column_with_min_data() if last_col is None else last_col

        if col.counter == 0:
            return

        row = col.down
        while row != col:
            row_num = row.counter
            if row_num < last_row_num:
                # already processed, skip it
                row = row.down
                continue
            row.cover_column(col)
            solution.append(row_num)
            next_cell = row.right
            while next_cell != row:
                next_cell.cover_column(next_cell.header)
                next_cell = next_cell.right
            if col.multiplicity == 0:
                self._explore(solution)
            else:
                self._explore(solution, col, row_num)
            next_cell = row.left
            while next_cell != row:
                next_cell.uncover_column(next_cell.header)
                next_cell = next_cell.left
            solution.pop()
            row.uncover_column(col)
            row = row.down
        return

    def _choose_column_with_min_data(self):
        col = self._sparse_matrix.right
        min_col = col
        min_val = col.counter

        while col != self._sparse_matrix:
            if col.counter < min_val:
                min_col = col
                min_val = col.counter
            col = col.right

        return min_col
