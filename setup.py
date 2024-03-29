from setuptools import setup, find_packages


setup_info = dict(
    name="model",
    version=1.0,
    packages=find_packages(),
    install_requires=[
        "gensim",
        "pandas",
        "polars",
        "coverage",
    ],
)

setup(**setup_info)
