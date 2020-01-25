<img src="docs/img/logo.png" width=15% align="center" />

# Dexterous Robot Hand

## Usage

### Requirements

This project was scripted and tested for `python 3.6` and should also work on any newer version of python.
Due to heavy usage of type hinting the code is not compatible with any older version of python!

To install all required python packages run

```bash
pip install -r requirements.txt
```

Additionally some libraries and permissions will be required. Run

```bash
sudo bash install.sh
```

To train on any mujoco-based environment, you will need to install the MuJoCo software (requiring a license) as well as mujoco-py.

To save gifs during training, imagemagicks policy needs to be changed to allow more memory usage. Achieve this by e.g. commenting out all ressource lines in `/etc/ImageMagick-6/policy.xml`. 

### Main Scripts

The python files `train.py` and `evaluate.py` provide ready-made scripts for training 
and evaluating an agent in an environment. With `pretrain.py`, it is possible to pretrain the visual component
of a network on classification or reconstruction.
