import itertools
from collections import OrderedDict
from typing import Dict
from typing import Tuple

import numpy as np
from gymnasium import spaces


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

    if len(adresses) == 0:
        return np.zeros(0), np.zeros(0)

    return model.names[adresses[0]:].split(b'\x00')[:len(adresses)], adresses


def mj_qpos_dict_to_qpos_vector(model, qpos_dict: Dict):
    """From a given dictionary of joint name-position pairs, create a vector using their order in the model."""
    qpos = np.array(
        [qpos_dict[n] for n in map(lambda x: x.decode("utf-8"), model.names.split(b"\x00")) if n in qpos_dict.keys()]
    )

    return qpos


_FLOAT_EPS = np.finfo(np.float64).eps
_EPS4 = _FLOAT_EPS * 4.0

# Copyright (c) 2009-2017, Matthew Brett and Christoph Gohlke
#    All rights reserved.
#
#    Redistribution and use in source and binary forms, with or without
#    modification, are permitted provided that the following conditions are
#    met:
#
#    1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
#    2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#
#    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
#    IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
#    THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
#    PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
#    CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
#    EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
#    PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
#    PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
#    LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
#    NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Many methods borrow heavily or entirely from transforms3d:
# https://github.com/matthew-brett/transforms3d
# They have mostly been modified to support batched operations.

"""
Rotations
=========

Note: these have caused many subtle bugs in the past.
Be careful while updating these methods and while using them in clever ways.

See MuJoCo documentation here: http://mujoco.org/book/modeling.html#COrientation

Conventions
-----------
    - All functions accept batches as well as individual rotations
    - All rotation conventions match respective MuJoCo defaults
    - All angles are in radians
    - Matricies follow LR convention
    - Euler Angles are all relative with 'xyz' axes ordering
    - See specific representation for more information

Representations
---------------

Euler
    There are many euler angle frames -- here we will strive to use the default
        in MuJoCo, which is eulerseq='xyz'.
    This frame is a relative rotating frame, about x, y, and z axes in order.
        Relative rotating means that after we rotate about x, then we use the
        new (rotated) y, and the same for z.

Quaternions
    These are defined in terms of rotation (angle) about a unit vector (x, y, z)
    We use the following <q0, q1, q2, q3> convention:
            q0 = cos(angle / 2)
            q1 = sin(angle / 2) * x
            q2 = sin(angle / 2) * y
            q3 = sin(angle / 2) * z
        This is also sometimes called qw, qx, qy, qz.
    Note that quaternions are ambiguous, because we can represent a rotation by
        angle about vector <x, y, z> and -angle about vector <-x, -y, -z>.
        To choose between these, we pick "first nonzero positive", where we
        make the first nonzero element of the quaternion positive.
    This can result in mismatches if you're converting an quaternion that is not
        "first nonzero positive" to a different representation and back.

Axis Angle
    (Not currently implemented)
    These are very straightforward.  Rotation is angle about a unit vector.

XY Axes
    (Not currently implemented)
    We are given x axis and y axis, and z axis is cross product of x and y.

Z Axis
    This is NOT RECOMMENDED.  Defines a unit vector for the Z axis,
        but rotation about this axis is not well defined.
    Instead pick a fixed reference direction for another axis (e.g. X)
        and calculate the other (e.g. Y = Z cross-product X),
        then use XY Axes rotation instead.

SO3
    (Not currently implemented)
    While not supported by MuJoCo, this representation has a lot of nice features.
    We expect to add support for these in the future.

TODO / Missing
--------------
    - Rotation integration or derivatives (e.g. velocity conversions)
    - More representations (SO3, etc)
    - Random sampling (e.g. sample uniform random rotation)
    - Performance benchmarks/measurements
    - (Maybe) define everything as to/from matricies, for simplicity
"""


def euler2mat(euler):
    """Convert Euler Angles to Rotation Matrix.  See rotation.py for notes"""
    euler = np.asarray(euler, dtype=np.float64)
    assert euler.shape[-1] == 3, "Invalid shaped euler {}".format(euler)

    ai, aj, ak = -euler[..., 2], -euler[..., 1], -euler[..., 0]
    si, sj, sk = np.sin(ai), np.sin(aj), np.sin(ak)
    ci, cj, ck = np.cos(ai), np.cos(aj), np.cos(ak)
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk

    mat = np.empty(euler.shape[:-1] + (3, 3), dtype=np.float64)
    mat[..., 2, 2] = cj * ck
    mat[..., 2, 1] = sj * sc - cs
    mat[..., 2, 0] = sj * cc + ss
    mat[..., 1, 2] = cj * sk
    mat[..., 1, 1] = sj * ss + cc
    mat[..., 1, 0] = sj * cs - sc
    mat[..., 0, 2] = -sj
    mat[..., 0, 1] = cj * si
    mat[..., 0, 0] = cj * ci
    return mat


def euler2quat(euler):
    """Convert Euler Angles to Quaternions.  See rotation.py for notes"""
    euler = np.asarray(euler, dtype=np.float64)
    assert euler.shape[-1] == 3, "Invalid shape euler {}".format(euler)

    ai, aj, ak = euler[..., 2] / 2, -euler[..., 1] / 2, euler[..., 0] / 2
    si, sj, sk = np.sin(ai), np.sin(aj), np.sin(ak)
    ci, cj, ck = np.cos(ai), np.cos(aj), np.cos(ak)
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk

    quat = np.empty(euler.shape[:-1] + (4,), dtype=np.float64)
    quat[..., 0] = cj * cc + sj * ss
    quat[..., 3] = cj * sc - sj * cs
    quat[..., 2] = -(cj * ss + sj * cc)
    quat[..., 1] = cj * cs - sj * sc
    return quat


def mat2euler(mat):
    """Convert Rotation Matrix to Euler Angles.  See rotation.py for notes"""
    mat = np.asarray(mat, dtype=np.float64)
    assert mat.shape[-2:] == (3, 3), "Invalid shape matrix {}".format(mat)

    cy = np.sqrt(mat[..., 2, 2] * mat[..., 2, 2] + mat[..., 1, 2] * mat[..., 1, 2])
    condition = cy > _EPS4
    euler = np.empty(mat.shape[:-1], dtype=np.float64)
    euler[..., 2] = np.where(
        condition,
        -np.arctan2(mat[..., 0, 1], mat[..., 0, 0]),
        -np.arctan2(-mat[..., 1, 0], mat[..., 1, 1]),
    )
    euler[..., 1] = np.where(
        condition, -np.arctan2(-mat[..., 0, 2], cy), -np.arctan2(-mat[..., 0, 2], cy)
    )
    euler[..., 0] = np.where(
        condition, -np.arctan2(mat[..., 1, 2], mat[..., 2, 2]), 0.0
    )
    return euler


def quat2euler(quat):
    """Convert Quaternion to Euler Angles.  See rotation.py for notes"""
    return mat2euler(quat2mat(quat))


def quat2mat(quat):
    """Convert Quaternion to rotation angle.  See rotation.py for notes"""
    quat = np.asarray(quat, dtype=np.float64)
    assert quat.shape[-1] == 4, "Invalid shape quat {}".format(quat)

    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    Nq = np.sum(quat * quat, axis=-1)
    s = 2.0 / Nq
    X, Y, Z = x * s, y * s, z * s
    wX, wY, wZ = w * X, w * Y, w * Z
    xX, xY, xZ = x * X, x * Y, x * Z
    yY, yZ, zZ = y * Y, y * Z, z * Z

    mat = np.empty(quat.shape[:-1] + (3, 3), dtype=np.float64)
    mat[..., 0, 0] = 1.0 - (yY + zZ)
    mat[..., 0, 1] = xY - wZ
    mat[..., 0, 2] = xZ + wY
    mat[..., 1, 0] = xY + wZ
    mat[..., 1, 1] = 1.0 - (xX + zZ)
    mat[..., 1, 2] = yZ - wX
    mat[..., 2, 0] = xZ - wY
    mat[..., 2, 1] = yZ + wX
    mat[..., 2, 2] = 1.0 - (xX + yY)
    return np.where((Nq > _FLOAT_EPS)[..., np.newaxis, np.newaxis], mat, np.eye(3))


def quat_conjugate(q):
    inv_q = -q
    inv_q[..., 0] *= -1
    return inv_q


def quat_mul(q0, q1):
    assert q0.shape == q1.shape
    assert q0.shape[-1] == 4
    assert q1.shape[-1] == 4

    w0 = q0[..., 0]
    x0 = q0[..., 1]
    y0 = q0[..., 2]
    z0 = q0[..., 3]

    w1 = q1[..., 0]
    x1 = q1[..., 1]
    y1 = q1[..., 2]
    z1 = q1[..., 3]

    w = w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1
    x = w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1
    y = w0 * y1 + y0 * w1 + z0 * x1 - x0 * z1
    z = w0 * z1 + z0 * w1 + x0 * y1 - y0 * x1
    q = np.array([w, x, y, z])
    if q.ndim == 2:
        q = q.swapaxes(0, 1)
    assert q.shape == q0.shape
    return q


def quat2axisangle(quat):
    theta = 0
    axis = np.array([0, 0, 1])
    sin_theta = np.linalg.norm(quat[1:])

    if sin_theta > 0.0001:
        theta = 2 * np.arcsin(sin_theta)
        theta *= 1 if quat[0] >= 0 else -1
        axis = quat[1:] / sin_theta

    return axis, theta


def get_parallel_rotations():
    mult90 = [0, np.pi / 2, -np.pi / 2, np.pi]
    parallel_rotations = []
    for euler in itertools.product(mult90, repeat=3):
        canonical = mat2euler(euler2mat(euler))
        canonical = np.round(canonical / (np.pi / 2))
        if canonical[0] == -2:
            canonical[0] = 2
        if canonical[2] == -2:
            canonical[2] = 2
        canonical *= np.pi / 2
        if all([(canonical != rot).any() for rot in parallel_rotations]):
            parallel_rotations += [canonical]
    assert len(parallel_rotations) == 24
    return parallel_rotations


def convert_observation_to_space(observation):
    if isinstance(observation, dict):
        space = spaces.Dict(
            OrderedDict(
                [
                    (key, convert_observation_to_space(value))
                    for key, value in observation.items()
                ]
            )
        )
    elif isinstance(observation, np.ndarray):
        low = np.full(observation.shape, -float("inf"), dtype=np.float32)
        high = np.full(observation.shape, float("inf"), dtype=np.float32)
        space = spaces.Box(low, high, dtype=observation.dtype)
    else:
        raise NotImplementedError(type(observation), observation)

    return space
