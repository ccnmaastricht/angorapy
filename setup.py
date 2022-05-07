from setuptools import setup, find_packages

setup(
    name='angorapy',
    version='0.7.0',
    description='ANthropomorphic Goal-ORiented Modeling, Learning and Analysis for Neuroscience',
    url='https://github.com/ccnmaastricht/dexterous-robot-hand',
    author='Tonio Weidler',
    author_email='t.weidler@maastrichtuniversity.nl',
    license='GPL-3.0',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy==1.19.2",
        "Box2D",
        "gym==0.18",
        "tensorflow==2.4.2",
        "mpi4py==3.1.3",
        "tqdm",
        "simplejson",
        "psutil",
        "scipy",
        "sklearn",
        "argcomplete",
        "matplotlib",
        "scikit-learn==0.24.1",
        "pandas==1.0.4",
        "nvidia-ml-py3",
        "seaborn",
        "distance",

        # webinterface
        "Flask~=1.1.2",
        "Jinja2==3.0.0",
        "bokeh",
        "flask_jsglue"
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