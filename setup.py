from setuptools import setup
from catkin_pkg.python_setup import generate_distutils_setup

# Add submodules as Python modules
setup_args = generate_distutils_setup(
    packages=['AB3DMOT_libs'],
    package_dir={'': 'src/AB3DMOT'})
setup(**setup_args)

setup_args = generate_distutils_setup(
    packages=['ab3dmot_ros'],
    package_dir={'': 'src'})
setup(**setup_args)

setup_args = generate_distutils_setup(
    packages=['xinshuo_io'],
    package_dir={'': 'src/Xinshuo_PyToolbox'})
setup(**setup_args)

setup_args = generate_distutils_setup(
    packages=['xinshuo_miscellaneous'],
    package_dir={'': 'src/Xinshuo_PyToolbox'})
setup(**setup_args)