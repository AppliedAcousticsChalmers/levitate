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
        return {'short': None}

    # Format the version
    description = p_out.stdout.decode().strip().lstrip('Vversion')
    parts = description.split('-')
    version = {'short': parts[0]}

    if len(parts) > 1:
        version['full'] = version['short'] + '+' + '.'.join(parts[1:])
    else:
        version['full'] = version['short']

    version['release'] = '.g' not in version['full']
    version['clean'] = 'dirty' not in version['full']

    try:
        p_out = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=d, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except subprocess.CalledProcessError:
        version['git revision'] = None
    else:
        version['git revision'] = p_out.stdout.decode().strip()

    return version


version_info = git_version()
version = version_info['short']


packaged_contents = \
f'''"""File generated while packaging."""
import contextlib

version_info = {version_info}
version_info['packaged'] = True
version = '{version}'


@contextlib.contextmanager
def hardcoded():
    """Dummy context manager, returns the version."""
    yield version
'''

version_info['packaged'] = False


@contextlib.contextmanager
def hardcoded():
    """Context manager for hardcoding the version while packaging.

    Use this context managed around the `setup()` call while packaging to replace
    the dynamic git-tag versioning with a hardcoded version number.
    Will return the current version, hardcode the file for packaging, and
    restore the file afterwards.
    """
    if version is None:
        raise RuntimeError('Cannot get version from git, nothing to hardcode')

    with open(__file__, 'r') as f:
        contents = f.read()
    with open(__file__, 'w') as f:
        f.write(packaged_contents)
    try:
        yield version
    finally:
        with open(__file__, 'w') as f:
            f.write(contents)
