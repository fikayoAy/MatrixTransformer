"""
MatrixTransformer Setup Configuration
A unified framework for structure-preserving matrix transformations
"""

from setuptools import setup, find_packages
import os

# Read the long description from README
def read_long_description():
    here = os.path.abspath(os.path.dirname(__file__))
    readme_path = os.path.join(here, 'README.md')
    
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ''

# Read requirements from requirements.txt
def read_requirements():
    here = os.path.abspath(os.path.dirname(__file__))
    req_path = os.path.join(here, 'requirements.txt')
    
    requirements = []
    if os.path.exists(req_path):
        with open(req_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if line and not line.startswith('#'):
                    requirements.append(line)
    
    return requirements

setup(
    name='matrixtransformer',
    version='0.1.0',
    author='Fikayomi Ayodele',
    author_email='Ayodeleanjola4@gmail.com',
    description='A deterministic AI framework for structure-preserving matrix transformations',
    long_description=read_long_description(),
    long_description_content_type='text/markdown',
    url='https://github.com/fikayoAy/MatrixTransformer',
    project_urls={
        'Bug Tracker': 'https://github.com/fikayoAy/MatrixTransformer/issues',
        'Documentation': 'https://github.com/fikayoAy/MatrixTransformer#readme',
        'Source Code': 'https://github.com/fikayoAy/MatrixTransformer',
        'Changelog': 'https://github.com/fikayoAy/MatrixTransformer/blob/main/CHANGELOG.md',
        'Related Project': 'https://github.com/fikayoAy/quantum_accel',
        'Paper - MatrixTransformer': 'https://zenodo.org/records/15867279',
        'Paper - Hyperdimensional': 'https://doi.org/10.5281/zenodo.16051260',
    },
    
    # Package discovery
    py_modules=['matrixtransformer', 'base', 'base_classes', 'graph'],
    packages=find_packages(exclude=['benchmarks', 'benchmarks.*', '__pycache__']),
    
    # Python version requirement
    python_requires='>=3.8',
    
    # Dependencies
    install_requires=[
        'numpy>=1.21.0',
        'pandas>=1.3.0',
        'scipy>=1.7.0',
        'scikit-learn>=1.0.0',
        'torch>=1.9.0',
        'networkx>=2.6',
        'matplotlib>=3.5.0',
        'pillow>=8.3.0',
        'tqdm>=4.62.0',
        'joblib>=1.0.0',
    ],
    
    # Optional dependencies
    extras_require={
        'full': read_requirements(),
        'viz': [
            'matplotlib>=3.5.0',
            'seaborn>=0.11.0',
            'plotly>=5.0.0',
        ],
        'ml': [
            'umap-learn>=0.5.0',
            'hdbscan>=0.8.0',
            'statsmodels>=0.13.0',
        ],
        'jupyter': [
            'jupyter>=1.0.0',
            'jupyterlab>=3.0.0',
            'ipywidgets>=7.6.0',
            'ipython>=7.0.0',
        ],
        'dev': [
            'pytest>=6.0.0',
            'black>=21.0',
            'flake8>=3.9.0',
            'mypy>=0.910',
        ],
    },
    
    # Package metadata
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    
    keywords=[
        'matrix-transformations',
        'linear-algebra',
        'deterministic-ai',
        'structure-preserving',
        'hyperdimensional-computing',
        'manifold-learning',
        'scientific-computing',
        'machine-learning',
        'quantum-inspired',
        'graph-transformations',
    ],
    
    # Include additional files
    include_package_data=True,
    package_data={
        '': ['*.txt', '*.md', 'LICENSE.txt'],
    },
    
    # Entry points (if you want CLI commands)
    # entry_points={
    #     'console_scripts': [
    #         'matrixtransformer=matrixtransformer:main',
    #     ],
    # },
    
    # Additional metadata
    zip_safe=False,
    platforms='any',
)
