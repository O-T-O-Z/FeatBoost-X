import pathlib

from setuptools import find_packages, setup

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / "README.md").read_text(encoding="utf-8")

with open("requirements.txt") as f:
    requirements = list(f.read().splitlines())

setup(
    name="featboostx",
    version="0.1.0rc2",
    description="A Python package for the FeatBoost-X feature selection algorithm",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/O-T-O-Z/FeatBoost-X",
    author="Ömer Tarik Özyilmaz & Ahmad Alsahaf",
    author_email="o.t.ozyilmaz@umcg.nl",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="Feature Selection, Gradient Boosting, FeatBoost-X",
    package_dir={"featboostx": "featboostx"},
    packages=find_packages(include=["featboostx", "featboostx.*"]),
    python_requires=">=3.10, <4",
    install_requires=requirements,
)
