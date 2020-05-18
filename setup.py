from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="vame",
    version='0.1',
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
    install_requires=[
        "pytest-shutil",
        "scipy<=1.2.1",
        "numpy",
        "matplotlib",
        "pathlib",
	"pandas",
        "ruamel.yaml",
	"sklearn",
        "pyyaml",
        "opencv-python",
    ],
)
