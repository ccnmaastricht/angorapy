from setuptools import setup

setup(
    name='dexterity',
    version='1.0.0',
    description='Anthropomorphic Robotic Control with Biologically Plausible Networks',
    url='https://github.com/ccnmaastricht/dexterous-robot-hand',
    author='Tonio Weidler',
    author_email='t.weidler@maastrichtuniversity.ml',
    license='GPL-3.0',
    packages=['dexterity'],
    install_requires=[
        "numpy==1.19.0",
        "gym==0.18",
        "tensorflow=2.4.0",
        "mpi4py=3.0.3",
        "tqdm",
        "simplejson",
        "psutil",
        "scipy",
        "sklearn",
        "argcomplete",
        "matplotlib",
        "scikit-learn==0.24.1",
        "pandas==1.0.4",
        "seaborn",
        "distance"
    ],

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