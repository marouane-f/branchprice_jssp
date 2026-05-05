# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False

"""
High-performance implementation of the shortest-path dynamic program.  
This module mirrors the behaviour of the Python reference implementation 
but leverages Cython/typed-memoryviews for speed.
"""

# cython: language_level=3
cimport cython
from libc.math cimport INFINITY
import numpy as np
cimport numpy as np
from utils import timing_store, timeit_accumulate, iter_store, iter_accumulate

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline double _comp_sum_rc_term(double[:, ::1] a,
                                     Py_ssize_t m,
                                     Py_ssize_t begin,
                                     Py_ssize_t end) nogil:
    cdef double s = 0.0
    cdef Py_ssize_t t
    for t in range(begin, end + 1):
        s += a[m, t]
    return s

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double comp_sum_rc_term(a,
                              Py_ssize_t m,
                              Py_ssize_t b,
                              Py_ssize_t e):
    cdef np.ndarray[np.float64_t, ndim=2] arr = np.asarray(a, dtype=np.float64, order="C")
    cdef double[:, ::1] mv = arr
    return _comp_sum_rc_term(mv, m, b, e)

@iter_accumulate(iter_store["shortest_path"])
@timeit_accumulate(timing_store["shortest_path"])
def compute_shortest_path(int n_machines,
                          int C_max,
                          dur,
                          seq,
                          rc_term,
                          *,
                          branch_operations=None,
                          branch_operations1=None,
                          branch_operations0=None):
    """
    Cython implementation of the dynamic program used in the pricing oracle.
    The arguments mirror the original Python function; the return value is the
    tuple ``(F, path)`` where ``F`` is the cost table and ``path`` contains the
    predecessor information (as tuples or ``None``).
    """
    
    cdef np.ndarray[np.float64_t, ndim=2] rc_arr = np.asarray(
        rc_term, dtype=np.float64, order="C")
    cdef Py_ssize_t n_m = rc_arr.shape[0]
    cdef Py_ssize_t horizon = rc_arr.shape[1]

    cdef np.ndarray[np.int64_t, ndim=1] dur_arr = np.asarray(dur, dtype=np.int64, order="C")
    cdef np.ndarray[np.int64_t, ndim=1] seq_arr = np.asarray(seq, dtype=np.int64, order="C")

    cdef bint seq_zero_based = np.min(seq_arr) == 0

    cdef np.ndarray[np.int64_t, ndim=1] q = np.zeros(n_m + 1, dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim=1] seq_vec = np.zeros(n_m + 1, dtype=np.int64)

    cdef Py_ssize_t idx
    for idx in range(n_m):
        q[idx + 1] = dur_arr[idx]
        seq_vec[idx + 1] = seq_arr[idx]

    cdef np.ndarray[np.float64_t, ndim=2] rc_padded = np.zeros((n_m + 1, horizon + 1), 
                                                                dtype=np.float64, order="C")
    for idx in range(n_m):
        rc_padded[idx + 1, 1:horizon + 1] = rc_arr[idx, :]

    cdef double[:, ::1] rc_view = rc_padded

    cdef np.ndarray[np.float64_t, ndim=2] F = np.empty(
        (n_m + 1, horizon + 1), dtype=np.float64)
    F.fill(INFINITY)
    F[0, :] = 0.0
    cdef double[:, :] F_view = F

    cdef np.ndarray[np.int64_t, ndim=2] prev_op = np.full(
        (n_m + 1, horizon + 1), -1, dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim=2] prev_time = np.full(
        (n_m + 1, horizon + 1), -1, dtype=np.int64)
    cdef long[:, :] prev_op_view = prev_op
    cdef long[:, :] prev_time_view = prev_time

    cdef tuple arc
    cdef int m0, t0
    cdef dict branch_map = {}
    cdef dict branch1_map = {}

    if branch_operations is not None:
        for arc in branch_operations:
            _, m0, t0 = arc
            m0 = int(m0)
            t0 = int(t0)
            branch_map.setdefault(m0 + 1, []).append(t0)
    if branch_operations1 is not None:
        for arc in branch_operations1:
            _, m0, t0 = arc
            m0 = int(m0)
            t0 = int(t0)
            branch1_map.setdefault(m0 + 1, []).append(t0)
    

    cdef Py_ssize_t i, k, t, row, m_idx, remaining, earl, lat
    cdef Py_ssize_t q_i
    cdef double first_term, second_term, second_term_sum, prev_sum
    cdef bint has_prev, arc_forbidden

    for i in range(1, n_m + 1):
        row = i
        m_idx = seq_vec[i]
        q_i = q[m_idx]

        earl = 0
        for k in range(1, i + 1):
            earl += q[seq_vec[k]]

        remaining = 0
        for k in range(i + 1, n_m + 1):
            remaining += q[seq_vec[k]]
        lat = horizon - remaining
        if lat <= earl:
            continue

        prev_sum = 0.0
        has_prev = False

        for t in range(earl, lat):
            first_term = INFINITY if t == 0 else F_view[row, t]

            arc_forbidden = False
            if not arc_forbidden and m_idx in branch_map:
                for t0 in branch_map[m_idx]:
                    if t0 <= t <= t0 + q_i - 1:
                        arc_forbidden = True
                        break
            if not arc_forbidden and m_idx in branch1_map:
                for t0 in branch1_map[m_idx]:
                    if t < t0 or t > t0 + q_i - 1:
                        arc_forbidden = True
                        break

            second_term = INFINITY
            if not arc_forbidden:
                if has_prev:
                    second_term_sum = (
                        prev_sum
                        - rc_view[m_idx, t - q_i + 1]
                        + rc_view[m_idx, t + 1]
                    )
                else:
                    # second_term_sum = _comp_sum_rc_term(rc_view, m_idx, t - q_i + 1, t)
                    second_term_sum = _comp_sum_rc_term(rc_view, m_idx, t - q_i + 2, t + 1)

                second_term = F_view[row - 1, t - q_i + 1] + second_term_sum
                prev_sum = second_term_sum
                has_prev = True
            else:
                has_prev = False

            if first_term <= second_term:
                F_view[row, t + 1] = first_term
                prev_op_view[row, t + 1] = i
                prev_time_view[row, t + 1] = t - 1
            else:
                F_view[row, t + 1] = second_term
                prev_op_view[row, t + 1] = i - 1
                prev_time_view[row, t + 1] = t - q_i

    F_out = np.asarray(F[:, 1:], order="C")
    path_out = []
    cdef Py_ssize_t col
    cdef list row_path
    for row in range(n_m + 1):
        row_path = []
        for col in range(1, horizon + 1):
            if prev_op_view[row, col] < 0:
                row_path.append(None)
            else:
                row_path.append(
                    (int(prev_op_view[row, col]), int(prev_time_view[row, col]))
                )
        path_out.append(row_path)

    return F_out, path_out
