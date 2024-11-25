# lab3_1_convolve.pyx

from cython.parallel import prange
import numpy as np
cimport numpy as np

def convolve_batch(batch, np.ndarray[DTYPE_t, ndim=2] image, np.ndarray[np.int32_t, ndim=2] kernel):
    cdef i, j
    for i, j in batch:
        value = np.sum(kernel * padded_image[i:i+kernel.shape[0], j:j+kernel.shape[1]])
        results.append((i, j, value))
