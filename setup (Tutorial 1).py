from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name='cppcuda_tutorial',
    version='0.0.1',
    author='qgli',
    description='A tutorial of cpp extension with cuda',
    ext_modules=[
        CppExtension(
            name='cppcuda_tutorial', 
            sources=['interpolation.cpp'])
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)