"""Backtracking Algorithms – N-Queens and Sudoku Solver
=======================================================
This module provides classic backtracking solutions:

1) N-Queens: place N queens on an N×N board so that no two queens attack.
   Returns all distinct board configurations.

2) Sudoku Solver: solve a 9×9 Sudoku puzzle (0 denotes empty cells).
   Uses constraint checks on rows, columns, and 3×3 subgrids.
"""

from __future__ import annotations
from typing import List, Tuple, Optional


class BacktrackingAlgorithms:
    """Collection of backtracking problems and solutions."""

    # ------------------------------- N-Queens ------------------------------
    def n_queens(self, n: int) -> Tuple[List[List[str]], int]:
        """Generate all solutions to the N-Queens problem.

        Returns (solutions, count), where each solution is represented as a
        list of strings, each string with exactly one 'Q' and otherwise '.'.

        Time: exponential; Space: O(n) recursion + O(n) sets
        """
        if n <= 0:
            return [], 0

        solutions: List[List[str]] = []
        cols = set()
        diag1 = set()  # r - c
        diag2 = set()  # r + c
        board = ["." * n for _ in range(n)]

        # To build rows efficiently, store columns of queens
        placement: List[int] = [-1] * n  # placement[row] = col

        def place(row: int) -> None:
            if row == n:
                # build board strings from placement
                sol: List[str] = []
                for r in range(n):
                    c = placement[r]
                    sol.append("." * c + "Q" + "." * (n - c - 1))
                solutions.append(sol)
                return
            for c in range(n):
                d1 = row - c
                d2 = row + c
                if c in cols or d1 in diag1 or d2 in diag2:
                    continue
                cols.add(c); diag1.add(d1); diag2.add(d2)
                placement[row] = c
                place(row + 1)
                placement[row] = -1
                cols.remove(c); diag1.remove(d1); diag2.remove(d2)

        place(0)
        return solutions, len(solutions)

    # ----------------------------- Sudoku Solver ---------------------------
    def sudoku_solve(self, grid: List[List[int]]) -> Optional[List[List[int]]]:
        """Solve a 9×9 Sudoku puzzle using backtracking.

        Input: grid with values 0..9 (0 means empty). Returns a solved grid
        or None if no solution exists. Does not mutate the input grid.
        """
        if len(grid) != 9 or any(len(row) != 9 for row in grid):
            raise ValueError("grid must be 9x9")

        # Work on a copy
        board = [row[:] for row in grid]

        def find_empty() -> Optional[Tuple[int, int]]:
            for r in range(9):
                for c in range(9):
                    if board[r][c] == 0:
                        return r, c
            return None

        def valid(r: int, c: int, val: int) -> bool:
            # Row and column
            if any(board[r][x] == val for x in range(9)):
                return False
            if any(board[x][c] == val for x in range(9)):
                return False
            # 3x3 box
            br, bc = 3 * (r // 3), 3 * (c // 3)
            for i in range(br, br + 3):
                for j in range(bc, bc + 3):
                    if board[i][j] == val:
                        return False
            return True

        def solve() -> bool:
            empty = find_empty()
            if not empty:
                return True
            r, c = empty
            for v in range(1, 10):
                if valid(r, c, v):
                    board[r][c] = v
                    if solve():
                        return True
                    board[r][c] = 0
            return False

        return board if solve() else None


if __name__ == "__main__":
    bt = BacktrackingAlgorithms()
    sols, cnt = bt.n_queens(4)
    print("n_queens(4) solutions:", cnt)
    sample = [
        [5,3,0,0,7,0,0,0,0],
        [6,0,0,1,9,5,0,0,0],
        [0,9,8,0,0,0,0,6,0],
        [8,0,0,0,6,0,0,0,3],
        [4,0,0,8,0,3,0,0,1],
        [7,0,0,0,2,0,0,0,6],
        [0,6,0,0,0,0,2,8,0],
        [0,0,0,4,1,9,0,0,5],
        [0,0,0,0,8,0,0,7,9],
    ]
    solved = bt.sudoku_solve(sample)
    print("sudoku solved:", solved is not None)
