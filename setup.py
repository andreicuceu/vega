#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages
import glob

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

scripts = glob.glob('bin/*')

requirements = ['numpy', 'scipy', 'astropy', 'numba', 'iminuit', 'h5py', 'mcfit',
                'setuptools', 'cachetools', 'matplotlib', 'gitpython']

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest>=3', ]

setup(
    author="Andrei Cuceu",
    author_email='andreicuceu@gmail.com',
    python_requires='>=3.7',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Astronomy',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9'
    ],
    description="Package for modeling and fitting 2-point statistics for the Lyman-alpha forest.",
    entry_points={
        'console_scripts': [
            'vega=vega.cli:main',
        ],
    },
    install_requires=requirements,
    license="GNU General Public License v3.0",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='vega',
    name='vega',
    packages=find_packages(include=['vega', 'vega.*']),
    setup_requires=setup_requirements,
    scripts=scripts,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/andreicuceu/Vega',
    version='0.3.0',
    zip_safe=False,
)
