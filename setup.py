"""The setup script."""
import sys
import pathlib
from setuptools import find_packages, setup

README = pathlib.Path("docs/index.md").read_text()
HISTORY = pathlib.Path("CHANGELOG.md").read_text()

REQUIREMENTS = ["prettyqt"]
version = ".".join(map(str, sys.version_info))

setup(
    author="Philipp Temminghoff",
    author_email="phil65@kodi.tv",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    # entry_points={
    #     "pyinstaller40": ["hook-dirs=cutepandas.__pyinstaller:get_hook_dirs"],
    # },
    description="Pythonic layer on top of PyQt5 / PySide2 / PySide6",
    install_requires=REQUIREMENTS,
    license="MIT license",
    python_requires=">=3.9.0",
    long_description=README + "\n\n" + HISTORY,
    long_description_content_type="text/markdown",
    include_package_data=True,
    keywords="cutepandas",
    name="cutepandas",
    packages=find_packages(),
    test_suite="tests",
    url="https://github.com/phil65/cutepandas",
    version="0.2.1",
    zip_safe=False,
)
