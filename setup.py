#!/usr/bin/env python3
"""embeddix setup.py.

This file details modalities for packaging the embeddix application.
"""

from setuptools import setup

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    name='embeddix',
    description='A general purpose toolkit for word embeddings',
    author=' Alexandre Kabbach',
    author_email='akb@3azouz.net',
    long_description=long_description,
    long_description_content_type='text/markdown',
    version='0.1.0',
    url='https://github.com/akb89/embeddix',
    download_url='https://github.com/akb89/embeddix',
    license='MIT',
    keywords=['word embeddings', 'random', 'hack'],
    platforms=['any'],
    packages=['embeddix', 'embeddix.logging', 'embeddix.exceptions',
              'embeddix.utils'],
    package_data={'embeddix': ['logging/*.yml', 'resources/*']},
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'embeddix = embeddix.main:main'
        ],
    },
    install_requires=['pyyaml>=4.2b1', 'tensorflow==1.15.2',
                      'bert-serving-client==1.10.0', 'tqdm==4.35.0'],
    classifiers=['Development Status :: 2 - Pre-Alpha',
                 'Environment :: Web Environment',
                 'Intended Audience :: Developers',
                 'Intended Audience :: Education',
                 'Intended Audience :: Science/Research',
                 'License :: OSI Approved :: MIT License',
                 'Natural Language :: English',
                 'Operating System :: OS Independent',
                 'Programming Language :: Python :: 3.5',
                 'Programming Language :: Python :: 3.6',
                 'Topic :: Scientific/Engineering :: Artificial Intelligence',
                 'Topic :: Software Development :: Libraries :: Python Modules'],
    zip_safe=False,
)
