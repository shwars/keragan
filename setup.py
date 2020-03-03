#!/usr/bin/env python

import setuptools
import keragan

with open('README.md') as readme_file:
    readme = readme_file.read()
    
setuptools.setup(
    name='keragan',
    packages=setuptools.find_packages(),
    version=keragan.__version__,
    install_requires=['keras','imutils','opencv-python','matplotlib'],
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