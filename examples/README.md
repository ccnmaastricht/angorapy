_this documentation is WIP_

# 🚀 Getting Started with AngoraPy

## Basic Training

## 🧰 Customized Usage
AngoraPy was built with modularity in mind. Many aspects of the training process can be customized through your own implementations. The most common customization is the usage of your own network architecture. The distributed reinforcement learning pipeline requires your model's implementation to follow specific rules. These are specified [here]() alongside a tutorial on how to incorporate your own model into the training process. The present README will introduce you to several other customizations (WIP), link examples and also provide directions to further details.

### Custom Policy Distributions
Currently, four builtin policy distributions are supported, the Gaussian and the Beta distribution for *continuous* and the (multi-)categorical distribution for *discrete* environments. 

To implement new policy distributions, extend the *BasePolicyDistribution* abstract class.

### Custom Environments

### Custom Reward Functions

### Custom Input and Output Transformers

### Custom Hyperparameters Configurations
Last, and somewhat least, you can add your own preset hyperparameter configuration. 

```python
from dexterity.configs.hp_config import make_config, derive_config

my_conf = make_config(
    batch_size=128,
    workers=8,
    model="lstm",
    architecture="wider"
)

my_sub_conf = derive_config(my_conf,
    {"model": "gru"}
)
```

Hyperparameter configurations created `make_config(...)` automatically assign default values to all required parameters and let you specify only those you want to change. With the help of `derive_config(...)` you can build variants of other configurations, including your own created by either `make_config(...)` or `derive_config(...)`.
