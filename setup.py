import os

from setuptools import setup, find_packages

from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()


setup(
    name='angorapy',
    version='0.10.0',
    description='ANthropomorphic Goal-ORiented Modeling, Learning and Analysis for Neuroscience',
    long_description_content_type='text/markdown',
    long_description=long_description,
    url='https://github.com/ccnmaastricht/angorapy',
    author='Tonio Weidler',
    author_email='research@tonioweidler.de',
    license='GPL-3.0',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "swig==4.1.1",
        "imageio==2.28.1",
        "numpy==1.23.5",

        # tensorflow and extensions
        "tensorflow==2.12.0",
        "tensorflow_probability==0.20.1",
        "tensorflow_graphics==2021.12.3",
        "mpi4py==3.1.4",
        "tqdm",
        "simplejson",
        "psutil",
        "scipy",
        "scikit-learn",
        "argcomplete",
        "matplotlib",
        "scikit-learn==0.24.1",
        "pandas==1.4.4",
        "nvidia-ml-py3",
        "seaborn",
        "distance",
        "panda_gym==3.0.6",
        "statsmodels==0.14.0",
        "keras_cortex==0.0.7",

        # environments
        "box2d-py==2.3.5",
        "gym[all]==0.26.2",
        "mujoco==2.2",
        "dm_control==1.0.12",

        # webinterface
        "itsdangerous==2.0.1",
        "werkzeug==2.0.3",
        "Flask~=1.1.2",
        "Jinja2==3.0.0",
        "bokeh==2.3.3",
        "flask_jsglue",
    ],

    package_data={
        "angorapy": ["environments/assets/**/*"],
    },

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
)
