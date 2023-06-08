from typing import Dict, Tuple

import numpy as np


def robot_get_obs(model, data):
    """Returns all joint positions and velocities associated with a robot."""
    joint_names, _ = mj_get_category_names(model, "jnt")
    if data.qpos is not None and joint_names:
        names = [n for n in joint_names if n.startswith(b"robot")]
        return (
            np.array([data.jnt(name).qpos for name in names]).flatten(),
            np.array([data.jnt(name).qvel for name in names]).flatten(),
        )
    return np.zeros(0), np.zeros(0)


def mj_get_category_names(model, category: str) -> Tuple[np.ndarray, np.ndarray]:
    """Returns the names and adresses of elements of a given category in the model.

    Args:
        model: The MuJoCo model to extract the information from.
        category: The category of elements to extract, e.g. jnt, actuator, tendon.

    Returns:
        A tuple of two numpy arrays, the first containing the names, the second the adresses.
    """

    adresses = getattr(model, f"name_{category}adr")
    return model.names[adresses[0]:].split(b'\x00')[:len(adresses)], adresses


def mj_qpos_dict_to_qpos_vector(model, qpos_dict: Dict):
    """From a given dictionary of joint name-position pairs, create a vector using their order in the model."""
    return np.array(
        [qpos_dict[n] for n in map(lambda x: x.decode("utf-8"), model.names.split(b"\x00")) if n in qpos_dict.keys()]
    )
