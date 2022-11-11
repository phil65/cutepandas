# cutepandas: Pythonic layer on top of PyQt5 / PySide2 / PySide6
[![PyPI Latest Release](https://img.shields.io/pypi/v/cutepandas.svg)](https://pypi.org/project/cutepandas/)
[![Package Status](https://img.shields.io/pypi/status/cutepandas.svg)](https://pypi.org/project/cutepandas/)
[![License](https://img.shields.io/pypi/l/cutepandas.svg)](https://github.com/phil65/cutepandas/blob/master/LICENSE)
[![Travis Build Status](https://travis-ci.org/phil65/cutepandas.svg?branch=master)](https://travis-ci.org/phil65/cutepandas)
[![CodeCov](https://codecov.io/gh/phil65/CutePandas/branch/master/graph/badge.svg)](https://codecov.io/gh/phil65/CutePandas)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PyUp](https://pyup.io/repos/github/phil65/CutePandas/shield.svg)](https://pyup.io/repos/github/phil65/CutePandas/)

## What is it?

**CutePandas** is a Python package that provides a pythonic layer on top of the GUI frameworks PyQt5 / PySide2 / PySide6.

## Main Features
Here are just a few of the things that CutePandas does well:

  - Large parts of the Qt API are available in a **PEP-8**-compliant way.
  - Several predefined widgets, validators, models, syntax highlighters are included.
  - A regex module based on QRegularExpression with the same API as Pythons core re module.


   [widgets]: https://phil65.github.io/CutePandas/widgets.html
   [validators]: https://phil65.github.io/CutePandas/validators.html
   [syntaxhighlighters]: https://phil65.github.io/CutePandas/syntaxhighlighters.html
   [models]: https://phil65.github.io/CutePandas/models.html


## Where to get it
The source code is currently hosted on GitHub at:
https://github.com/phil65/CutePandas

The latest released version are available at the [Python
package index](https://pypi.org/project/cutepandas).

```sh
# or PyPI
pip install cutepandas
```

## Dependencies
- [bidict](https://pypi.org/project/bidict)
- [orjson](https://pypi.org/project/orjson)
- [regex](https://pypi.org/project/regex)
- [docutils](https://pypi.org/project/docutils)


## Installation from sources

In the `cutepandas` directory (same one where you found this file after
cloning the git repo), execute:

```sh
python setup.py install
```

or for installing in [development mode](https://pip.pypa.io/en/latest/reference/pip_install.html#editable-installs):


```sh
python -m pip install -e .
```

## License
[MIT](LICENSE)

## Documentation
The official documentation is hosted on Github Pages: https://phil65.github.io/CutePandas/

## Contributing to cutepandas [![Open Source Helpers](https://www.codetriage.com/phil65/cutepandas/badges/users.svg)](https://www.codetriage.com/phil65/cutepandas)

All contributions, bug reports, bug fixes, documentation improvements, enhancements, and ideas are welcome.

Or maybe through using CutePandas you have an idea of your own or are looking for something in the documentation and thinking ‘this can be improved’...you can do something about it!
