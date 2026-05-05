# setup.py
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        name="shortest_path._shortest_path_c",            # package.module
        sources=["shortest_path/_shortest_path_c.pyx"],
        include_dirs=[np.get_include()],
        language="c",
    )
]

setup(
    name="shortest-path",
    version="0.1.0",
    ext_modules=cythonize(extensions, compiler_directives={
        "boundscheck": False,
        "wraparound": False,
        "initializedcheck": False,
        "cdivision": True,
    }),
    zip_safe=False,
)