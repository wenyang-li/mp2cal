from setuptools import setup
import glob
import os.path as path
from os import listdir

__version__ = '0.0.0'

setup_args = {
    'name': 'mp2cal',
    'author': 'Wenyang Li',
    'url': 'https://github.com/wenyang-li/mp2cal.git',
    'description': 'MWA Phase II calibration',
    'package_dir': {'mp2cal': 'src'},
    'packages': ['mp2cal'],
#    'scripts': glob.glob('scripts/*'),
    'version': __version__,
    # 'package_data':
}


if __name__ == '__main__':
    apply(setup, (), setup_args)
