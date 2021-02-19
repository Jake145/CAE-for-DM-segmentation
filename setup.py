from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='CAE-Jake_HP_145',
    version='1.3',
    author='Jake Harold Pensavalle',
description="CAE helper functions, models and classes",url='https://github.com/Jake145/CAE-for-DM-segmentation',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author_email='jakeharold.pensavalle@gmail.com',classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(),
    python_requires='>=3.8',
    )