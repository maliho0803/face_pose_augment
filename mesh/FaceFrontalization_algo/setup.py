'''
python setup.py build_ext -i
to compile
'''

# setup.py
from distutils.core import setup, Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy
import sys
sys.path.append('.')

setup(
	name = 'zbuffer_core_cython',
    cmdclass={'build_ext': build_ext},
    ext_modules=[Extension("FaceFrontalization_core_cython",
                 sources=["FaceFrontalization_core_cython.pyx", "FF_FaceFrontalization.cpp", "FF_MeshSrc2Ref.cpp"],
                 language='c++',
                 include_dirs=[numpy.get_include()],
                 extra_compile_args=['-stdlib=libc++'],
                 extra_link_args=['-stdlib=libc++'])],
)