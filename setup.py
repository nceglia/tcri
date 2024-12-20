from setuptools import setup, find_packages

setup(
    name='tcri',
    version='0.0.1',
    description='Information theoretic metrics for single cell RNA and TCR sequencing.',
    packages=find_packages(include=['tcri','tcri.metrics','tcri.preprocessing','tcri.plotting','tcri.utils','tcri.model']),
    #install_requires=["scipy","numpy","notebook","sklearn","pandas","scanpy","tqdm","seaborn","matplotlib","pysankey"],
)
