# conflict_lb.pyx
# cython: boundscheck=False, wraparound=False, nonecheck=False, language_level=3
cimport numpy as cnp
cimport cython

cdef inline bint _all_le_ub_idx_list(object idx_list,
                                     cnp.intp_t[::1] counts,
                                     Py_ssize_t ub):
    cdef Py_ssize_t i, n = len(idx_list), idx
    for i in range(n):
        idx = <Py_ssize_t> idx_list[i]
        if counts[idx] > ub:
            return False
    return True

cpdef object compute_conflict_lb_untimed(cnp.intp_t[::1] remaining_jobs,  # np.intp from np.argsort
                                         object          all_jobs,        # list/seq; all_jobs[j] -> cols
                                         object          forbidden_mt,    # set of (m,t)
                                         cnp.intp_t[::1] usage_counts,    # np.intp
                                         Py_ssize_t      max_mach):
    cdef Py_ssize_t n_all = len(all_jobs)
    cdef Py_ssize_t n_rem = remaining_jobs.shape[0]
    cdef bint jobs_at_max_mach = (n_all - n_rem) >= max_mach
    cdef Py_ssize_t max_mach_1 = max_mach - 1
    cdef double total_lb = 0.0
    cdef object is_disjoint = forbidden_mt.isdisjoint
    cdef Py_ssize_t r, job
    cdef object curr_cols, col

    for r in range(n_rem):
        job = remaining_jobs[r]
        curr_cols = all_jobs[job]
        for col in curr_cols:  # assumed sorted by col.cost asc.
            if not is_disjoint(col.mt_pairs):
                continue
            if _all_le_ub_idx_list(col.time_steps_flat_list, usage_counts, max_mach_1):
                total_lb += col.cost
                break
            if not jobs_at_max_mach:
                total_lb += col.cost
                break
        else:
            return None
    return total_lb
