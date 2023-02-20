from setuptools import setup, find_packages

setup(
    name="pytorch2jax",
    version="0.1",
    packages=find_packages(),
    python_requires=">=3.6, <4",
    install_requires=["torch", "jax", "jaxlib"],
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    author="Your Name",
    description="Convert PyTorch models to Jax functions and Flax models",
    url="https://github.com/yourusername/pytorch2jax",
)
