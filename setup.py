from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pytorch2jax",
    version="0.1.1",
    packages=find_packages(),
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.6, <4",
    install_requires=["torch", "jax", "jaxlib"],
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    author="Subhojeet Pramanik",
    description="Convert PyTorch models to Jax functions and Flax models",
    url="https://github.com/subho406/Pytorch2Jax",
)
