from setuptools import find_packages, setup

setup(
    name='phlame',
    version='1.0.1',
    packages=find_packages(),
    description='Novelty-aware intraspecies profiling of metagenome samples',
    url='https://github.com/quevan/phlame',
    author='Evan Qu',
    author_email='equ@mit.edu',
    license='MIT',
    scripts=['bin/phlame'],
    install_requires=[
    'numpy',
    'pandas',
    'seaborn',
    'matplotlib',
    'biopython',
    'ete3',
    'statsmodels',
    'pytest'
    ],

)
