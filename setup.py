from setuptools import setup, find_packages

__version__ = 'unknown'
for line in open('levitate/__init__.py'):
    if line.startswith('__version__'):
        exec(line)
        break


setup(
    name='levitate',
    version=__version__,
    description='Python implementations from the Levitate research project',
    long_description=open('README.rst').read(),
    long_description_content_type='text/x-rst',
    url='https://github.com/AppliedAcousticsChalmers/levitate',
    author='Carl Andersson',
    author_email='carl.andersson@chalmers.se',
    license='MIT',
    packages=find_packages('.'),
    python_requires='>=3.5',
    install_requires=[
        'numpy',
        'scipy'],
    include_package_data=True,
)
