from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as req_file:
    requirements = req_file.read().splitlines()


setup(
    name="vame",
    version='1.0',
    packages=find_packages(),
    entry_points={"console_scripts": "vame = vame:main"},
    author="K. Luxem & P. Bauer",
    description="Variational Animal Motion Embedding.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://https://github.com/LINCellularNeuroscience/VAME/",
    setup_requires=[
        "pytest",
    ],	
    install_requires=requirements,
)
