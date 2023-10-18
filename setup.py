import glob
import os.path as osp
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

ROOT_DIR = osp.dirname(osp.abspath(__file__))
include_dirs = [osp.join(ROOT_DIR, 'include')]

# list includes all .cpp and .cu files in the current directory for compilation
sources = glob.glob('*.cpp')+glob.glob('*.cu') 

setup(
    name='cppcuda_tutorial',
    version='0.0.1',
    author='qgli',
    description='A tutorial of cpp extension with cuda',
    ext_modules=[
        CUDAExtension(
            name='cppcuda_tutorial',
            sources=sources,
            include_dirs=include_dirs,
            extra_compile_args={'cxx': ['-O2'], # O2 is for reducing the size of the compiled file
                                'nvcc': ['-O2']}
        ) 
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)