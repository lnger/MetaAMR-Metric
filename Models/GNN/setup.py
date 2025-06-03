# setup.py (修改后)
from setuptools import setup, Extension  # 改为setuptools
from Cython.Build import cythonize
import numpy

setup(
    name='encode_seq',
    ext_modules=cythonize(
        Extension(
            'encode_seq',
            sources=['encode_seq.pyx'],
            include_dirs=[numpy.get_include()],  # 保留numpy依赖
        )
    ),
)