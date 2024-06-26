from setuptools import setup, find_packages

setup(
    name='harmonic_feature_selection',
    version='0.1.1',
    description='Code for Harmonic Feature Selection paper',
    author='Otto Sumray',
    author_email='sumray@maths.ox.ac.uk',
    packages=find_packages(),
    install_requires=[
        'ordered_set',
        'scikit-learn',
        'scikit-sparse',
        'numpy',
        'scipy'
    ],
)
