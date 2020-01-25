<img src="docs/img/logo.png" width=15% align="right" />

# Dexterous Robot Hand

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

## Usage

The python files `train.py` and `evaluate.py` provide ready-made scripts for training 
and evaluating an agent in any environment. With `pretrain.py`, it is possible to pretrain the visual component
of a network on classification or reconstruction. `benchmark.py` provides functionality for training a batch of agents 
possibly using different configs for comparison of strategies.

### Training an Agent

By using the `train.py` script you can train an agent on any environment and set hyperparameters, model and other
 options. Additionally, a monitor will be automatically added to the drilling of the agent, s.t. you can inspect
 the training progress. For more detail consult the monitoring <a href="monitor/README.md">README</a>. 

The scripts commandline parameters are as follows:

```
usage: train.py [{any registered environment}]
                [-h] [--model {ffn,rnn,lstm,gru}]
                [--distribution {categorical,gaussian,beta}] [--shared]
                [--iterations ITERATIONS] [--config CONFIG] [--cpu]
                [--load-from LOAD_FROM] [--preload PRELOAD]
                [--export-file EXPORT_FILE] [--eval] [--save-every SAVE_EVERY]
                [--monitor-frequency MONITOR_FREQUENCY]
                [--gif-every GIF_EVERY] [--debug] [--workers WORKERS]
                [--horizon HORIZON] [--discount DISCOUNT] [--lam LAM]
                [--no-state-norming] [--no-reward-norming] [--epochs EPOCHS]
                [--batch-size BATCH_SIZE] [--lr-pi LR_PI]
                [--lr-schedule {None,exponential}] [--clip CLIP]
                [--c-entropy C_ENTROPY] [--c-value C_VALUE] [--tbptt TBPTT]
                [--grad-norm GRAD_NORM] [--clip-values] [--stop-early]

positional arguments:
  {... any registered environment} the target environment

optional arguments:
  -h, --help            show this help message and exit
  --model {ffn,rnn,lstm,gru}
                        model type if not shadowhand
  --distribution {categorical,gaussian,beta}
  --shared              make the model share part of the network for policy
                        and value
  --iterations ITERATIONS
                        number of iterations before training ends
  --config CONFIG       config name (utilities/configs.py) to be loaded
  --cpu                 use cpu only
  --load-from LOAD_FROM
                        load from given agent id
  --preload PRELOAD     load visual component weights from pretraining
  --export-file EXPORT_FILE
                        save policy to be loaded in workers into file
  --eval                evaluate separately (instead of using worker
                        experience)
  --save-every SAVE_EVERY
                        save agent every given number of iterations
  --monitor-frequency MONITOR_FREQUENCY
                        update the monitor every n iterations.
  --gif-every GIF_EVERY
                        make a gif every n iterations.
  --debug               run in debug mode
  --workers WORKERS     the number of workers exploring the environment
  --horizon HORIZON     the number of optimization epochs in each cycle
  --discount DISCOUNT   discount factor for future rewards
  --lam LAM             lambda parameter in the GAE algorithm
  --no-state-norming    do not normalize states
  --no-reward-norming   do not normalize rewards
  --epochs EPOCHS       the number of optimization epochs in each cycle
  --batch-size BATCH_SIZE
                        minibatch size during optimization
  --lr-pi LR_PI         learning rate of the policy
  --lr-schedule {None,exponential}
                        lr schedule type
  --clip CLIP           clipping range around 1 for the objective function
  --c-entropy C_ENTROPY
                        entropy factor in objective function
  --c-value C_VALUE     value factor in objective function
  --tbptt TBPTT         length of subsequences in truncated BPTT
  --grad-norm GRAD_NORM
                        norm for gradient clipping, 0 deactivates
  --clip-values         clip value objective
  --stop-early          stop early if threshold of env was surpassed
```