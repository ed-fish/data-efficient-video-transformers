from setuptools import setup, find_packages

setup(
    name="src",
    version="0.0.1",
    description="Frame stacked transformer test",
    author="Ed Fish",
    author_email="edward.fish@surrey.ac.uk",
    url="https://github.com/ed-fish/self-supervised-video",
    install_requires=["pytorch-lightning"],
    packages=find_packages(),
)