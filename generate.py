import sys

from crossword import *


class CrosswordCreator():

    def __init__(self, crossword):
        """
        Create new CSP crossword generate.
        """
        self.crossword = crossword
        self.domains = {
            var: self.crossword.words.copy()
            for var in self.crossword.variables
        }

    def letter_grid(self, assignment):
        """
        Return 2D array representing a given assignment.
        """
        letters = [
            [None for _ in range(self.crossword.width)]
            for _ in range(self.crossword.height)
        ]
        for variable, word in assignment.items():
            direction = variable.direction
            for k in range(len(word)):
                i = variable.i + (k if direction == Variable.DOWN else 0)
                j = variable.j + (k if direction == Variable.ACROSS else 0)
                letters[i][j] = word[k]
        return letters

    def print(self, assignment):
        """
        Print crossword assignment to the terminal.
        """
        letters = self.letter_grid(assignment)
        for i in range(self.crossword.height):
            for j in range(self.crossword.width):
                if self.crossword.structure[i][j]:
                    print(letters[i][j] or " ", end="")
                else:
                    print("█", end="")
            print()

    def save(self, assignment, filename):
        """
        Save crossword assignment to an image file.
        """
        from PIL import Image, ImageDraw, ImageFont
        cell_size = 100
        cell_border = 2
        interior_size = cell_size - 2 * cell_border
        letters = self.letter_grid(assignment)

        # Create a blank canvas
        img = Image.new(
            "RGBA",
            (self.crossword.width * cell_size,
             self.crossword.height * cell_size),
            "black"
        )
        font = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 80)
        draw = ImageDraw.Draw(img)

        for i in range(self.crossword.height):
            for j in range(self.crossword.width):

                rect = [
                    (j * cell_size + cell_border,
                     i * cell_size + cell_border),
                    ((j + 1) * cell_size - cell_border,
                     (i + 1) * cell_size - cell_border)
                ]
                if self.crossword.structure[i][j]:
                    draw.rectangle(rect, fill="white")
                    if letters[i][j]:
                        w, h = draw.textsize(letters[i][j], font=font)
                        draw.text(
                            (rect[0][0] + ((interior_size - w) / 2),
                             rect[0][1] + ((interior_size - h) / 2) - 10),
                            letters[i][j], fill="black", font=font
                        )

        img.save(filename)

    def solve(self):
        """
        Enforce node and arc consistency, and then solve the CSP.
        """
        self.enforce_node_consistency()
        self.ac3()
        return self.backtrack(dict())

    def enforce_node_consistency(self):
        """
        Update `self.domains` such that each variable is node-consistent.
        (Remove any values that are inconsistent with a variable's unary
         constraints; in this case, the length of the word.)
        """
        self.domains = {
            v: set(x for x in self.domains[v] if len(x) == v.length)
            for v in self.domains
        }

    def revise(self, x, y):
        """
        Make variable `x` arc consistent with variable `y`.
        To do so, remove values from `self.domains[x]` for which there is no
        possible corresponding value for `y` in `self.domains[y]`.

        Return True if a revision was made to the domain of `x`; return
        False if no revision was made.
        """
        overlap = self.crossword.overlaps[x, y]
        if overlap is None:
            return False

        overlap_x, overlap_y = overlap
        init_length = len(self.domains[x])

        self.domains[x] = set(
            word_x for word_x in self.domains[x]
            if set(
                word_y for word_y in self.domains[y]
                if word_y[overlap_y] == word_x[overlap_x] and word_y != word_x
            ) != set()
        )
        return len(self.domains[x]) != init_length

    def ac3(self, arcs=None):
        """
        Update `self.domains` such that each variable is arc consistent.
        If `arcs` is None, begin with initial list of all arcs in the problem.
        Otherwise, use `arcs` as the initial list of arcs to make consistent.

        Return True if arc consistency is enforced and no domains are empty;
        return False if one or more domains end up empty.
        """
        queue = set(
            (x, y) for x in self.domains
            for y in self.crossword.neighbors(x)
        ) if arcs is None else set(arcs)

        while queue != set():
            x, y = queue.pop()
            if self.revise(x, y):
                if self.domains[x] == set():
                    return False
                for z in (self.crossword.neighbors(x) - {y}):
                    queue.add((z, x))
        return True

    def assignment_complete(self, assignment):
        """
        Return True if `assignment` is complete (i.e., assigns a value to each
        crossword variable); return False otherwise.
        """
        if len(self.crossword.variables) != len(assignment.keys()):
            return False

        for variable in self.crossword.variables:
            if variable not in assignment.keys():
                return False
        return True

    def consistent(self, assignment):
        """
        Return True if `assignment` is consistent (i.e., words fit in crossword
        puzzle without conflicting characters); return False otherwise.
        """
        if len(assignment) != len(set(assignment.values())):
            return False  # not all values are distinct

        for v1 in assignment:
            for v2 in {
                key: val
                for key, val in assignment.items()
                if key != v1
            }:
                overlap = self.crossword.overlaps[v1, v2]
                if overlap is None:
                    continue

                i, j = overlap
                if assignment[v1][i] != assignment[v2][j]:
                    return False

        return True

    def order_domain_values(self, var, assignment):
        """
        Return a list of values in the domain of `var`, in order by
        the number of values they rule out for neighboring variables.
        The first value in the list, for example, should be the one
        that rules out the fewest values among the neighbors of `var`.
        """
        track_num_of_ruled_out = {}
        for val in self.domains[var]:
            if val in assignment.values():
                continue
            track_num_of_ruled_out[val] = 0
            for neighbor in self.crossword.neighbors(var):
                overlap = self.crossword.overlaps[var, neighbor]
                for neighbor_val in self.domains[neighbor]:
                    if neighbor_val in assignment.values():
                        continue
                    i, j = overlap
                    if val[i] != neighbor_val[j]:
                        track_num_of_ruled_out[val] = track_num_of_ruled_out[val] + 1

        return [
            key
            for key, _ in sorted(track_num_of_ruled_out.items(), key=lambda item: item[1])
        ]

    def select_unassigned_variable(self, assignment):
        """
        Return an unassigned variable not already part of `assignment`.
        Choose the variable with the minimum number of remaining values
        in its domain. If there is a tie, choose the variable with the highest
        degree. If there is a tie, any of the tied variables are acceptable
        return values.
        """
        unassigned_domains = {
            var: self.domains[var]
            for var in self.domains
            if var in (self.crossword.variables - set(assignment.keys()))
        }

        if len(unassigned_domains) == 1:
            return list(unassigned_domains.keys())[0]

        min_num_of_vals_in_domain = len(
            min(
                unassigned_domains.values(),
                key=lambda val: len(val)
            )
        )

        vars_with_min_num_of_remaining_vals = {
            var
            for var in unassigned_domains
            if len(unassigned_domains[var]) == min_num_of_vals_in_domain
        }

        if len(vars_with_min_num_of_remaining_vals) == 1:
            return vars_with_min_num_of_remaining_vals[0]

        return max(
            vars_with_min_num_of_remaining_vals,
            key=lambda var:
                len(self.crossword.neighbors(var) - set(assignment.keys()))
        )

    def backtrack(self, assignment):
        """
        Using Backtracking Search, take as input a partial assignment for the
        crossword and return a complete assignment if possible to do so.

        `assignment` is a mapping from variables (keys) to words (values).

        If no assignment is possible, return None.
        """
        raise NotImplementedError


def main():

    # Check usage
    if len(sys.argv) not in [3, 4]:
        sys.exit("Usage: python generate.py structure words [output]")

    # Parse command-line arguments
    structure = sys.argv[1]
    words = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) == 4 else None

    # Generate crossword
    crossword = Crossword(structure, words)
    creator = CrosswordCreator(crossword)
    assignment = creator.solve()

    # Print result
    if assignment is None:
        print("No solution.")
    else:
        creator.print(assignment)
        if output:
            creator.save(assignment, output)


if __name__ == "__main__":
    main()
