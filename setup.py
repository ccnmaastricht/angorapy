import os

from setuptools import setup, find_packages

from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()


setup(
    name='angorapy',
    version='0.9.0',
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
        "imageio",
        "numpy==1.19.2",
        "box2d-py",
        "gym==0.26.2",
        "mujoco==2.2.2",
        "tensorflow==2.4.2",
        "tensorflow_probability==0.12.2",
        "mpi4py==3.1.3",
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
        "protobuf==3.19.0",
        "panda_gym",
        
        "keras_cortex==0.0.7",

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
