# setup.py
from setuptools import setup, find_packages
import os

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="chess-analyzer",
    version="2.0.0",
    author="Chess Analyzer Team",
    author_email="info@chessanalyzer.com",
    description="Advanced chess analysis tool with Stockfish integration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/chess-analyzer",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "chess-analyzer=main:main",
        ],
    },
    package_data={
        "chess_analyzer": [
            "assets/*.png",
            "assets/*.wav", 
            "assets/*.ogg",
            "assets/*.bin"
        ],
    },
    include_package_data=True,
)