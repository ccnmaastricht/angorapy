# The Task API

AngoraPy's task API is a collection of tools for building and interacting with anthropomorphic task environments. Additionally,
it provides a set of builtin tasks.

The focus of this API are anthropomorphic tasks, which are tasks that are designed to be solved by anthropomorphic bodies in the form of simulated robots. While they are its focus, the task API is not limited to anthropomorphic tasks. It is designed to be flexible enough to support a wide range of tasks, including cognitive tasks, and provides wrappers that can encapsulate any Gym(nasium) environment.

A side note: The terms _task_ and _environment_ are often used interchangeably. In this document and the general nomenclature of AngoraPy, _task_ refers to the goal that the body is trying to achieve, and _environment_ refers to the world that the body and task are placed in. As such the environment by itself is agnostic to its task, but it sets the constraints that the body must work within to achieve its task.

## Available Tasks


| Task Group | Body | Variants |
| ---------- | ---- | -------- |
| Reach      | Hand |          |
| FreeReach  | Hand |          |
| Manipulate | Hand |          |

## Anthropomorphic Environments

### World Building

Anthropomorphic environments are environments that can host tasks designed to be solved by an anthropomorphic body. We define them to be composed of a body, external objects, and the world these are placed in. AngoraPy offers the _PhysicalWorld_ interface to standardize and facilitate the creation of worlds. This physical world consists of a _robot_, which is the body the brain component of the agent controls, optional _external objects_ that the body can interact with, and the _stage_ they are placed in. The (physical) environment can however also be build entirely without this interface. In any case, certain standards need to be followed to allow the task API to correctly identify the separate components of any anthropomorphic environment.

To build any world model, and its components, AngoraPy relies on MuJoCo. MuJoCo models are defined in the MJCF format and can be build statically in XML or programmatically using dm_control's mjcf module. For either, consult the [MuJoCo (XML) documentation](https://mujoco.readthedocs.io/en/latest/modeling.html), and for the latter additionally [this guide](https://github.com/deepmind/dm_control/blob/main/dm_control/mjcf/README.md).

#### Modeling with PhysicalWorld

The PhysicalWorld interface allows a streamlined, simple standard to define, reuse and recombine components of anthropomorphic environments.

##### The Robot

Every world _must_ contain _exactly one_ robot entity. The actuators of this robot will be controlled by the brain component of the agent. All the components of the robot must be named according to 

A minimal robot could be defined as follows:

TODO

##### External Objects

External objects are entities that are not controlled by the agent's brain component, but can be interacted with by the agent's body. There can be any number of external objects in a world, including none.

##### The Stage

In a world's _Stage_, the robot and external objects are placed.

#### Modeling in XML and Other Ways

# The Wrapping System

AngoraPy uses wrappers to standardize the communication between its training components and any given Gymnasium or AngoraPy task. Wrappers are additionally used to inject behaviour into tasks that may be useful for any given application and not task specific.

## Registering and Making Tasks
