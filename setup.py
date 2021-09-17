import os
import pathlib

import pkg_resources
from setuptools import find_packages, setup

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DESCRIPTION = "X-MAGICAL is a benchmark suite for cross-embodiment visual imitation."
TEST_REQUIREMENTS = [
    "pytest-xdist",
    "pytype",
]
DEV_REQUIREMENTS = [
    "ipdb",
    "jupyter",
    "black",
    "isort",
    "flake8",
]
with pathlib.Path("requirements.txt").open() as requirements_txt:
    CORE_REQUIREMENTS = [
        str(requirement)
        for requirement in pkg_resources.parse_requirements(requirements_txt)
    ]


def readme():
    """Load README for use as package's long description."""
    with open(os.path.join(THIS_DIR, "README.md"), "r") as fp:
        return fp.read()


def get_version():
    locals_dict = {}
    with open(os.path.join(THIS_DIR, "xmagical", "version.py"), "r") as fp:
        exec(fp.read(), globals(), locals_dict)
    return locals_dict["__version__"]


setup(
    name="x-magical",
    version=get_version(),
    author="Kevin Zakka",
    license="ISC",
    description=DESCRIPTION,
    python_requires=">=3.8",
    long_description=readme(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=CORE_REQUIREMENTS,
    extras_require={
        "dev": [*DEV_REQUIREMENTS, *TEST_REQUIREMENTS],
        "test": TEST_REQUIREMENTS,
    },
    url="https://github.com/kevinzakka/x-magical/",
)
