import numpy as np
cimport numpy as np

DTYPE = np.float32
ctypedef np.float32_t DTYPE_t

def convolve(np.ndarray[DTYPE_t, ndim=2] image, np.ndarray[np.int32_t, ndim=2] kernel):
    cdef int kernel_size = kernel.shape[0]
    cdef int pad_width = kernel_size // 2
    cdef int image_height = image.shape[0]
    cdef int image_width = image.shape[1]
    cdef i, j

    output_image = np.zeros_like(image, dtype=np.float32)

    for i in range(pad_width, image_height - pad_width):
        for j in range(pad_width, image_width - pad_width):
            output_image[i, j] = np.sum(kernel * image[i - pad_width:i + pad_width + 1, j - pad_width:j + pad_width + 1])

    return output_image
