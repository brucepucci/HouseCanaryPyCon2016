"""
Bruce Pucci
Data Scientist, Progressive Insurance

Python 3.5.1 :: Anaconda 2.4.0 (64-bit)



############## Quick Explanation ####################

Lets simplify the problem to find the largest sub-array of length k 
    (where k > 1) in a larger array (m).

Suppose m = [0 1 2 3 4 5 6 7 8 9]

If we cumsum along m we get [0 1 3 6 10 15 21 28 36 45].

We can find the largest sub-array by taking the differences of the values k indices apart.

Suppose k = 2

m_rght_offset = [0 1 3 6 10 15 21 28 36 45 0  0]
m_left_offset = [0 0 0 1 3  6  10 15 21 28 36 45]

argmax(m_rght_offset - m_left_offset) = 9
So largest sub array is m[8:10]

This solution is similar but in 2 dimensions.

####################################################






Boilerplate code for your convenience. Feel free to modify as you want.

Testing your implemention:
    cat inputs/input0.txt | python solution.py
    cat inputs/input1.txt | python solution.py
    cat inputs/input2.txt | python solution.py
    cat inputs/input3.txt | python solution.py

    Check if your output matches the corresponding outputs in 'outputs' folder.
"""

from sys import stdin
import numpy as np


def _cumsum_matrix(m):
    return m.cumsum(axis=0).cumsum(axis=1)


def _shift_top(M):
    M = np.concatenate([np.array([[0]*M.shape[1]]), M], axis=0)
    return M


def _shift_left(M):
    M = np.concatenate([np.array([[0]*M.shape[0]]).T, M], axis=1)
    return M


def find_densest(big, small_ncols, small_nrows):
    big_nrows, big_ncols = big.shape

    if small_nrows > big_nrows or small_ncols > big_ncols:
        return

    cumsum_big = _cumsum_matrix(big)

    bottom_right = cumsum_big[small_nrows-1:, small_ncols-1:]
    bottom_left = _shift_left(cumsum_big[small_nrows-1:, :-small_ncols:])
    top_left = _shift_top(cumsum_big[:-small_nrows, :-small_ncols])
    top_left = _shift_left(top_left)
    top_right = _shift_top(cumsum_big[:-small_nrows:, small_ncols-1:])

    diff_matrix = bottom_right - bottom_left - top_right + top_left  # adding top left reverts the double counting of top left.
    return diff_matrix.max()


def read_problem_instances(ifile):
    """Reads problem instances from given file.

    See readme.md for file format.

    Returns:
        A generator that gives a tuple of (2D array, RECTANGLE_ROWS, RECTANGLE_COLUMNS)

    This function has been changed for performance reasons.
    """
    lines = ifile.read().splitlines()
    lines = (np.fromstring(x, dtype='int', sep=' ') for x in lines)
    N = next(lines)[0]
    for i in range(N):
        small_nrows, small_ncols = tuple(next(lines))
        big_nrows, big_ncols = tuple(next(lines))
        big = np.concatenate([next(lines) for x in range(big_nrows)]).reshape(big_nrows, big_ncols)
        yield big, small_nrows, small_ncols

if __name__ == '__main__':
    for big, small_nrows, small_ncols in read_problem_instances(stdin):
        print(find_densest(big, small_nrows=small_nrows, small_ncols=small_ncols))
