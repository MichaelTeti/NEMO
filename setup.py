from setuptools import setup

setup(
    name = 'nemo',
    version = '0.0.1',
    description = 'Neural Encoding Models for Ophys Data project module.',
    author = 'Michael Teti',
    author_email = 'mteti@fau.edu',
    packages = ['nemo'],
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
        'torch==1.7.1',
        'torchvision==0.8.2',
        'urllib3',
        'chardet',
        'idna<3',
        'jinja2<2.12.0',
        'oct2py'
    ]
)
