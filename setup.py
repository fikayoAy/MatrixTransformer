import setuptools
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="MatrixTransformer",
    version="0.1.0",
    author="fikayoAy",
    author_email="author@example.com",  # Replace with your email
    description="A unified framework for structure-preserving matrix transformations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fikayoAy/MatrixTransformer",  # Replace with your repository URL
    project_urls={
        "Bug Tracker": "https://github.com/fikayoAyMatrixTransformer/issues",
        "Documentation": "https://github.com/fikayoAy/MatrixTransformer#readme",
        "Related Project": "https://github.com/fikayoAy/quantum_accel",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    package_dir={"": "."},
    packages=find_packages(where="."),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.6.0",
        "torch>=1.9.0",
        "scikit-learn>=0.24.0",
    ],
    keywords=[
        "matrix-transformations",
        "linear-algebra",
        "matrix-operations",
        "scientific-computing",
        "machine-learning",
        "quantum-simulation",
    ],
)
