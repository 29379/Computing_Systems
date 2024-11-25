from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy as np

ext_modules = [
    Extension(
        "lab3_0_convolve",
        ["lab3_0_convolve.pyx"],
        include_dirs=[np.get_include()],
    ),
    Extension(
        "lab3_1_convolve",
        ["lab3_1_convolve.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
    )
]

setup(
    name='convolve-parallel-world',
    ext_modules=cythonize(ext_modules),
)
