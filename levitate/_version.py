import subprocess
import os.path
import contextlib


def git_version():
    """Get the package version from git tags."""
    d = os.path.dirname(__file__)
    cmd = ['git', 'describe', '--tags', '--dirty', '--always']
    try:
        p_out = subprocess.run(cmd, cwd=d, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except subprocess.CalledProcessError:
        return None

    # Format the version
    description = p_out.stdout.decode().strip().lstrip('Vversion')
    parts = description.split('-')
    version = parts[0]
    if len(parts) > 1:
        version += '+' + '.'.join(parts[1:])
    return version


__version__ = git_version()
distribute_contents = \
'''"""File generated while packaging."""
import contextlib
__version__ = '{}'


@contextlib.contextmanager
def hardcoded():
    """Dummy context manager, returns the version."""
    yield __version__
'''.format(__version__)


@contextlib.contextmanager
def hardcoded():
    """Context manager for hardcoding the version while packaging.

    Use this context managed around the `setup()` call while packaging to replace
    the dynamic git-tag versioning with a hardcoded version number.
    Will return the current version, hardcode the file for packaging, and
    restore the file afterwards.
    """
    if __version__ is None:
        raise RuntimeError('Cannot get version from git, nothing to hardcode')

    with open(__file__, 'r') as f:
        contents = f.read()
    with open(__file__, 'w') as f:
        f.write(distribute_contents)
    try:
        yield __version__
    finally:
        with open(__file__, 'w') as f:
            f.write(contents)
