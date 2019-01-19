import glob

__version__ = '0.0.0'

setup_args = {
    'name': 'mp2cal',
    'author': 'Wenyang Li',
    'author_email': 'wenyang_li at brown.edu',
    'url': 'https://github.com/wenyang-li/mp2cal.git',
    'description': 'MWA Phase II calibration',
    'package_dir': {'mp2cal': 'src'},
    'packages': ['mp2cal'],
    'scripts': glob.glob('scripts/*'),
    'version': __version__,
}


if __name__ == '__main__':
    from distutils.core import setup
    apply(setup, (), setup_args)
