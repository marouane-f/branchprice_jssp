# usage_ext.pyx
# cython: language_level=3
cimport cython
import numpy as np
cimport numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline bint _kernel_i32(Py_ssize_t[::1] idx,
                             np.int32_t[::1] uc,
                             int max_active):
    cdef Py_ssize_t k, n = idx.shape[0], i
    for k in range(n):
        i = idx[k]
        if uc[i] >= max_active:
            return True
    for k in range(n):
        i = idx[k]
        uc[i] += 1
    return False

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline bint _kernel_i64(Py_ssize_t[::1] idx,
                             np.int64_t[::1] uc,
                             int max_active):
    cdef Py_ssize_t k, n = idx.shape[0], i
    for k in range(n):
        i = idx[k]
        if uc[i] >= max_active:
            return True
    for k in range(n):
        i = idx[k]
        uc[i] += 1
    return False

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef bint update_usage_with_check_untimed(t_idx, usage_counts, int max_active):
    # indices (np.intp) → Py_ssize_t memoryview
    cdef np.ndarray idx_arr = np.asarray(t_idx, dtype=np.intp, order="C")
    cdef Py_ssize_t[::1] idx = idx_arr

    # usage array: require C-contiguous 1D and int32 or int64
    cdef np.ndarray uc_arr = np.asarray(usage_counts, order="C")
    if uc_arr.ndim != 1 or not uc_arr.flags["C_CONTIGUOUS"]:
        raise ValueError("usage_counts must be 1D and C-contiguous")

    cdef np.int32_t[::1] uc32
    cdef np.int64_t[::1] uc64

    if uc_arr.dtype == np.int32:
        uc32 = uc_arr
        return _kernel_i32(idx, uc32, max_active)
    elif uc_arr.dtype == np.int64:
        uc64 = uc_arr
        return _kernel_i64(idx, uc64, max_active)
    else:
        raise TypeError("usage_counts dtype must be int32 or int64")