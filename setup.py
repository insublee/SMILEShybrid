from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='SMILEShybrid',
    install_requires=required,
    packages=find_packages(),
    package_data={},
)
