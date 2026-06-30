"""
Setup script for Traffic State Discovery package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="traffic-state-discovery",
    version="1.0.0",
    description="Unsupervised Traffic State Discovery Using Multi-Feature Vehicle Analysis from Urban CCTV Streams",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Research Team",
    author_email="research@example.com",
    url="https://github.com/research/traffic-state-discovery",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        "ultralytics>=8.1.0",
        "opencv-python>=4.8.1",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "scikit-learn>=1.3.0",
        "scipy>=1.10.0",
        "shapely>=2.0.0",
        "tqdm>=4.65.0",
        "networkx>=3.0",
        "filterpy>=1.4.5",
        "pyyaml>=6.0",
        "python-dotenv>=1.0.0",
        "colorlog>=6.7.0",
        "rich>=13.0.0",
        "typing-extensions>=4.8.0",
    ],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)