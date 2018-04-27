from distutils.core import setup, Extension
from Cython.Build import cythonize

extensions = [
    Extension("kmeanscython.kmeansinit_cython", ["kmeanscython/kmeansinit_cython.pyx"]),
    Extension("kmeanscython.kmeansinitpp_cython", ["kmeanscython/kmeansinitpp_cython.pyx"]),
    Extension("kmeanscython.kmeansinitvv_cython", ["kmeanscython/kmeansinitvv_cython.pyx"]),
    Extension("kmeanscython.kmeansloop_cython", ["kmeanscython/kmeansloop_cython.pyx"]),
    "kmeanscython/__init__.pyx",
]

setup(name = "kmeansmethods",
      version = "1.0",
      author='Qi Gao',
      author_email='qi.gao@duke.edu',
      url='https://github.com/qigao1/sta663.git',
      packages = ['kmeanspython'],
      ext_modules = cythonize(extensions),
      )
