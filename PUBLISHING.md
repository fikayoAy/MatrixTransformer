# Publishing to PyPI

Follow these steps to publish the MatrixTransformer package to PyPI:

## Prerequisites

Make sure you have the required tools:

```bash
pip install --upgrade pip
pip install --upgrade build twine
```

## Building the package

From the root directory of the project, run:

```bash
python -m build
```

This will create the distribution packages in the `dist/` directory.

## Testing the package locally (optional)

You can install the package locally for testing:

```bash
pip install -e .
```

## Uploading to PyPI



### Testing on TestPyPI first (recommended)

Upload to TestPyPI to make sure everything works:

```bash
python -m twine upload --repository testpypi dist/*
```

You can then install from TestPyPI:

```bash
pip install --index-url https://test.pypi.org/simple/ matrixtransformer
```

### Uploading to the real PyPI

When you're confident everything works:

```bash
python -m twine upload dist/*
```

You'll be prompted for your PyPI username and password.

## After publishing

Your package will be available to install via:

```bash
pip install matrixtransformer
```

## Updating the package

When you want to release a new version:

1. Update the version number in `setup.py` and `__init__.py`
2. Build the package again
3. Upload to PyPI

Remember to follow [semantic versioning](https://semver.org/) for your version numbers.
