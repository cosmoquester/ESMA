from setuptools import find_packages, setup

setup(
    name="esma",
    version="0.0.1",
    description="This repository is template for my python project.",
    python_requires=">=3.7",
    install_requires=[],
    url="https://github.com/cosmoquester/esma.git",
    author="Sangjun Park",
    packages=find_packages(exclude=["tests"]),
)
