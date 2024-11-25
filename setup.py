from setuptools import setup, find_packages

setup(
    name='RAISING',
    version='1.0.0',
    author='Devashish Tripathi',
    author_email='devashishtripathi697@gmail.com',
    description='RAISING: A supervised deep learning framework for hyperparameter tuning and feature selection',
    long_description="This is a simple text",
    long_description_content_type='text/markdown',  # Specify content type as Markdown
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
    ]
)
