from setuptools import setup, find_packages

setup(
    name='macau',
    version='0.99.0',
    license='GNU GENERAL PUBLIC LICENSE Version 3',
    description='Uncertainty and Novelty Modelling for LightGBM',
    author='Janne Merilinna',
    author_email='jmerilinna@gmail.com',
    url='https://github.com/jmerilinna/macau',
    packages=find_packages(),
    install_requires=[
    			'joblib',
    			'lightgbm',
			'numpy',
			'pandas',
			'properscoring',
			'scikit_learn',
			'scipy',
			'tqdm'
    ],
)
