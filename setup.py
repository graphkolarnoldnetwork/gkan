import setuptools

# Load the long_description from README.md
with open("README.md", "r", encoding="utf8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pygkan",
    version="0.0.5",
    author="Mehrdad Kiamari",
    author_email="kiamari@usc.edu",
    description="Graph Kolmogorov Arnold Networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    # url="https://github.com/ANRGUSC/GKAN/",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
