from setuptools import setup
from catkin_pkg.python_setup import generate_distutils_setup

# Add submodules as Python modules
setup_args = generate_distutils_setup(
    packages=['AB3DMOT_libs'],
    package_dir={'': 'src/AB3DMOT'})
setup(**setup_args)

# Add submodules as Python modules
setup_args = generate_distutils_setup(
    packages=['ab3dmot_ros'],
    package_dir={'': 'src'})
setup(**setup_args)