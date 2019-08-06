from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['tensorflow-probability==0.5.0']


setup(
    name='module',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='Testing TFP.')
