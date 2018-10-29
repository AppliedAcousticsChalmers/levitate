from setuptools import setup

__version__ = 'unknown'
for line in open('levitate/__init__.py'):
    if line.startswith('__version__'):
        exec(line)
        break


setup(
    name='levitate',
    version=__version__,
    description='Python implementations from the Levitate research project',
    packages=['levitate'],
    install_requires=[
        'numpy',
        'scipy'],
)
