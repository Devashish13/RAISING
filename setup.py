from setuptools import setup, find_packages

setup(
    name='RAISING',
    version='1.0.1',
    author='Devashish Tripathi',
    author_email='devashishtripathi697@gmail.com',
    description='RAISING: A supervised deep learning framework for hyperparameter tuning and feature selection',
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.5.2",
        "numpy>=1.23.5",
        "tensorflow<2.16.0",
        "keras>=2.11.0",
        "scikit-learn>=1.0.2",
        "keras-tuner>=1.1.3",
        "pysnptools>=0.5.10",
        "tensorflow_addons>=0.18.0",
        "imbalanced-learn>=0.11.0",
        "shap>=0.41.0",
        "statsmodels>=0.14.0",
        "matplotlib>=3.6.2"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",  # Updated license
        "Operating System :: OS Independent",
    ],
    license="GPL-3.0-or-later",
    python_requires='>=3.9',
)
