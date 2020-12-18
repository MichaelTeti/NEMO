from setuptools import setup

setup(
    name = 'NEMO',
    version = '0.0.1',
    description = 'Neural Encoding Models for Ophys Data project module.',
    author = 'Michael Teti',
    author_email = 'mteti@fau.edu',
    packages = ['NEMO'],
    install_requires = [
        'numpy<1.19.0',
        'scipy',
        'allensdk',
        'progressbar2',
        'opencv-python',
        'sklearn',
        'matplotlib',
        'seaborn',
        'pandas==0.25.3',
        'h5py<3.0.0',
        'torch',
        'torchvision',
        'urllib3',
        'chardet',
        'idna',
        'jinja2<2.12.0',
        'oct2py'
    ]
)
