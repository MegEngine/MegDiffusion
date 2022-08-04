import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setuptools.setup(
    name = "megdiffusion",
    version = "0.0.2",
    author = "Chai",
    description = "MegEngine implementation of Diffusion Models",
    long_description = long_description,
    long_description_content_type="text/markdown",
    packages = setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)