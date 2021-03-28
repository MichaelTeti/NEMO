from setuptools import setup

setup(
    name = 'nemo',
    version = '0.0.1',
    description = 'Neural Encoding Models for Ophys Data project module.',
    author = 'Michael Teti',
    author_email = 'mteti@fau.edu',
    packages = ['nemo'],
    install_requires = [
        'allensdk==2.4.1',
        'chardet<4.0',
        'h5py<3.0.0',
        'idna<3',
        'jinja2<2.12.0',
        'matplotlib',
        'neptune-client',
        'numpy<1.19.0',
        'oct2py',
        'opencv-python',
        'pandas==0.25.3',
        'progressbar2',
        'psutil',
        'pytorch-lightning',
        'scipy',
        'seaborn',
        'sklearn',
        'torch==1.7.1',
        'torchvision==0.8.2',
        'urllib3'
    ]
)
