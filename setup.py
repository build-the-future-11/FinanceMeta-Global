from setuptools import setup, find_packages

setup(
    name="fi-jepa",
    version="0.1.0",
    description="Financial Informed JEPA",
    packages=find_packages(),
    install_requires=[
        "torch>=2.1.0",
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "pyyaml",
        "tqdm",
        "matplotlib",
        "einops",
    ],
)