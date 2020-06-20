import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="virgmo", # Replace with your own username
    version="0.1",
    author="Iurii Mozzhorin",
    author_email="iurii.mozzhorin@gmail.com",
    description="Variational inference for random graph models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mozzhorin/VIRGMo",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=['numpy', 'torch>=1.4.0', 'scipy>=1.0.0', 'numpy']
)