# setup.py
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np


ext = Extension(
    name="usage_ext.usage_ext",
    sources=["usage_ext/usage_ext.pyx"],
    include_dirs=[np.get_include()],
    language="c",
)

setup(
    name="usage_ext",
    ext_modules=cythonize([ext], compiler_directives={"language_level": 3}),
)