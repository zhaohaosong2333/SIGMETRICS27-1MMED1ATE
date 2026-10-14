"""Matrix operations utilities."""

from typing import List

import numpy as np


def mm_mul(a: List[List[int]], b: List[List[int]]) -> List[List[int]]:
    """Matrix multiplication over GF(2) (binary field, XOR-AND).

    In this compiler `mm_mul` is only ever applied to 0/1 matrices (one-hot
    reorder / selection matrices), so the GF(2) product
        c[i][j] = XOR_k ( a[i][k] AND b[k][j] )
    is exactly (A @ B) mod 2. Vectorized with numpy: the original pure-python
    triple loop is O(rows*inner*cols) and becomes intractable for large models
    (e.g. GBDT). The result is bit-identical to the loop for 0/1 inputs; this is
    a plaintext compile-time speedup only and does not touch the FHE runtime or
    the compiled model values.
    """
    if len(a[0]) != len(b):
        raise ValueError(f"Matrix dimension mismatch: A is {len(a)}x{len(a[0])}, "
                        f"B is {len(b)}x{len(b[0])}")

    # float64 goes through BLAS (int64 matmul does not, and is ~100x slower at forest scale).
    # Exact: operands are 0/1, so every partial sum is an integer far below 2**53.
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    return (np.rint(a_arr @ b_arr).astype(np.int64) % 2).tolist()


def diagonalize_matrix(m: List[List[int]]) -> List[List[int]]:
    """Diagonalize matrix: dm[j][i] = m[i][(i + j) % colnum] (colnum diagonals of rownum slots)."""
    arr = np.asarray(m, dtype=np.int64)
    rownum, colnum = arr.shape
    rows = np.arange(rownum)
    cols = (np.arange(colnum)[:, None] + rows[None, :]) % colnum
    return arr[rows[None, :], cols].tolist()


def plan_column_tiling(rows: int, cols: int, row_slots: int):
    """Split a rows x cols diagonal MatMul into K column blocks of equal width c = ceil(cols / K) so that
    each block's replicated input vector fits one rotation row: copies(rows, c) * c <= row_slots, with
    copies = max(2, ceil((rows + c - 1) / c)). Returns the smallest such (K, c), or None if no column
    split fits (rows alone exceed the row, which would need row tiling as well)."""
    for k in range(1, cols + 1):
        c = -(-cols // k)
        copies = max(2, -(-(rows + c - 1) // c))
        if copies * c <= row_slots:
            return k, c
        if c == 1:
            break
    return None
