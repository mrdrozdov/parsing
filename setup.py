"""
Simple check list from AllenNLP repo: https://github.com/allenai/allennlp/blob/master/setup.py
To create the package for pypi.
1. Change the version in __init__.py, setup.py as well as docs/source/conf.py.
2. Commit these changes with the message: "Release: VERSION"
3. Add a tag in git to mark the release: "git tag VERSION -m'Adds tag VERSION for pypi' "
   Push the tag to git: git push --tags origin master
4. Build both the sources and the wheel. Do not change anything in setup.py between
   creating the wheel and the source distribution (obviously).
   For the wheel, run: "python setup.py bdist_wheel" in the top level directory.
   (this will build a wheel for the python version you use to build it).
   For the sources, run: "python setup.py sdist"
   You should now have a /dist directory with both .whl and .tar.gz source versions.
5. Check that everything looks correct by uploading the package to the pypi test server:
   twine upload dist/* -r pypitest
   (pypi suggest using twine as other methods upload files via plaintext.)
   Check that you can install it in a virtualenv by running:
   pip install -i https://testpypi.python.org/pypi transformers
6. Upload the final version to actual pypi:
   twine upload dist/* -r pypi
7. Copy the release notes from RELEASE.md to the tag in github once everything is looking hunky-dory.
"""

import shutil
from pathlib import Path

from setuptools import find_packages, setup


extras = {}

extras["mecab"] = ["mecab-python3"]
extras["sklearn"] = ["scikit-learn"]
extras["torch"] = ["torch"]

extras["serving"] = ["pydantic", "uvicorn", "fastapi"]
extras["all"] = extras["serving"] + ["torch"]

extras["testing"] = ["pytest", "pytest-xdist"]
extras["quality"] = ["black", "isort", "flake8"]
extras["docs"] = ["recommonmark", "sphinx", "sphinx-markdown-tables", "sphinx-rtd-theme"]
extras["dev"] = extras["testing"] + extras["quality"] + ["mecab-python3", "scikit-learn", "torch"]

setup(
    name="parsing",
    version="0.0.1",
    author="Andrew Drozdov",
    author_email="andrew@mrdrozdov.com",
    description="Implementations of Natural Language Parsers",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="NLP parsing",
    license="Apache",
    url="https://github.com/mrdrozdov/parsing",
    package_dir={"": "src"},
    packages=find_packages("src"),
    install_requires=[
        "numpy",
        # accessing files from S3 directly
        "boto3",
        # filesystem locks e.g. to prevent parallel downloads
        "filelock",
        # for downloading models over HTTPS
        "requests",
        # progress bars in model download and training scripts
        "tqdm",
        # for OpenAI GPT
        "regex != 2019.12.17",
    ],
    extras_require=extras,
    scripts=[],
    python_requires=">=3.5.0",
    classifiers=[],
)