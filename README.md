# Dexterous Robot Hand

### Requirements

This project was scripted and tested for `python 3.6` and should also work on any newer version of python.
Due to heavy usage of type hinting the code is not compatible with any older version of python!

To install all required python packages run

```bash
pip install -r requirements.txt
```

While this might not be necessary for most uses, when you want to use the keras model plotting, run

```bash
sudo apt-get install graphviz
```

under any Debian-based Linux distribution.

### Main Scripts

The python files `train.py` and `evaluate.py` provide ready-made scripts for training 
and evaluating an agent in an environment. With `pretrain.py`, it is possible to pretrain the visual component
of a network on classification or reconstruction.