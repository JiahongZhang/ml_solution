# -*- coding: utf-8 -*-
import setuptools

PACKAGE_NAME = "ml_solution"

__version__ = None
with open(f"{PACKAGE_NAME}/__version__.py") as vf:
    exec(vf.read()) # get __version__

with open('README.md', 'r') as rf:
    long_description = rf.read()

required = [
    'pandas>=2.0',
    'torch>=1.12',
    'seaborn>=0.13',
    'timm>=0.6',
    'opencv-python>=4.8',
    'transformers>=4.36',
    'torchscale>=0.3',
    'scikit-learn>=1.3',
    'wandb>=0.16',
    'sentencepiece>=0.1.99'
    ]


setuptools.setup(
    name=PACKAGE_NAME,
    version=__version__,
    author="hugo",
    author_email="hugo279@foxmail.com",
    description="A mechine learning pipeline lib.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JiahongZhang/ml_solution",
    packages=setuptools.find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=required,
    python_requires='>=3.10'
)
