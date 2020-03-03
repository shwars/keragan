#!/usr/bin/env python

import setuptools

with open('README.md') as readme_file:
    readme = readme_file.read()
    
with open('requirements.txt') as req_file:
    req = req_file.readlines()

with open('keragan/__init__.py') as py_file:
    v = next(filter(lambda x: x.startswith('__version__'),py_file.readlines()))
    version = v.split(' = ')[1].strip()[1:-1]

setuptools.setup(
    name='keragan',
    packages=setuptools.find_packages(),
    version=version,
    install_requires=req,
    description='Keras GAN Library',
    author='Dmitri Soshnikov',
    author_email='dmitri@soshnikov.com',
    url='https://github.com/shwars/keragan',
    long_description=readme,
    long_description_content_type='text/markdown; charset=UTF-8',
    license='MIT license',
    classifiers=[
        "Programming Language :: Python :: 3",
#        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries :: Python Modules"
    ]
)