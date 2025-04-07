"""
#       Install Project Requirements 
"""
from setuptools import setup, find_packages

setup(
    name="ssl_training",
    url="",
    description="SSL training environment",
    packages=[
        package for package in find_packages() if package.startswith(("playgrounds", "agents"))
    ],
    install_requires=[
        "gymnasium >= 1.0.0",
        "torch",
        "tqdm",
        "stable_baselines3[extra]",
        "tensorboard"
    ],
)
