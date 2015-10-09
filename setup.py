try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'name': 'digidemod',
    'version': '0.1.3',
    'packages': ['digidemod'],
    'install_requires': ['numpy', 'scipy', 'matplotlib', 'nose', 'filters'],
    'author': 'Evan M. Davis',
    'author_email': 'emd@mit.edu',
    'url': '',
    'description': 'Python tools for the demodulation of digital signals.'
}

setup(**config)
