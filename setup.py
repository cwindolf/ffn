try:
    from setuptools import setup
except ModuleNotFoundError:
    from distutils.core import setup


setup(
    name='ffn',
    version='0.1.0',
    author='Michal Januszewski',
    author_email='mjanusz@google.com',
    packages=[
        'ffn', 'ffn.inference', 'ffn.training', 'ffn.utils', 'ffn.slurm'
    ],
    scripts=[
        'build_coordinates.py',
        'compute_partitions.py',
        'run_inference.py',
        'train.py',
    ],
    entry_points={
        'console_scripts': ['ngvis=ffn.utils.ngvis:main'],
    },
    url='https://github.com/google/ffn',
    license='LICENSE',
    description='Flood-Filling Networks for volumetric instance segmentation',
    long_description=open('README.md').read(),
    install_requires=[
        'scikit-image',
        'scipy',
        'numpy',
        'h5py',
        'pillow',
        'absl-py',
    ],
)
