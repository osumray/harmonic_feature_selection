from setuptools import setup, find_packages

setup(
    name='harmonic_feature_selection',
    version='0.1',
    description='Code for Harmonic Feature Selection paper',
    author='Otto Sumray',
    author_email='osumray@gmail.com',
    packages=find_packages(),
    install_requires=[
        'ordered_set',
        'scikit-learn',
        'scikit-sparse',
        'numpy',
        'scipy'
    ],
)
