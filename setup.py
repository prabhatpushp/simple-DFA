from setuptools import setup, find_packages

setup(
    name="dfa_models",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
    ],
    author="Your Name",
    description="A clean implementation of Deterministic Finite Automata",
    python_requires=">=3.7",
) 