from setuptools import setup, find_packages

setup(
    name='crc_crps',
    packages=find_packages(),
    version='0.1.0',
    description='CRC and CRPS Detection algorithm used in the paper "Sleep-stage dependence and co-existence of cardio-respiratory coordination and phase synchronization"',
    install_requires=['numpy', 'scipy', 'numba']
)