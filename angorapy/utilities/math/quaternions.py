from angorapy.tasks.utils import quat_conjugate
from angorapy.tasks.utils import quat_mul

import numpy as np


def rotate(quaternion, rotation_quaternion):
    """
    Rotate a quaternion by another quaternion.

    Args:
        quaternion (np.ndarray): The quaternion to rotate.
        rotation_quaternion (np.ndarray): The quaternion to rotate by.

    Returns:
        np.ndarray: The rotated quaternion.
    """

    return multiply(multiply(rotation_quaternion, quaternion), conjugate(rotation_quaternion))


def conjugate(quat):
    """Get the conjugate of a quaternion."""
    conj_quat = -quat
    conj_quat[..., 0] *= -1
    return conj_quat


def multiply(q0, q1):
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


def from_angle_and_axis(angle, axis):
    """Get the quaternion representation of a rotation around an axis by an angle.

    Args:
        angle (float): The angle in radians.
        axis (np.ndarray): The axis of rotation of shape (3,).
    """

    assert axis.shape == (3,)
    axis /= np.linalg.norm(axis)
    quat = np.concatenate([[np.cos(angle / 2.)], np.sin(angle / 2.) * axis])
    quat /= np.linalg.norm(quat)
    return quat


def to_axis_angle(quat):
    theta = 0
    axis = np.array([0, 0, 1])
    sin_theta = np.linalg.norm(quat[1:])

    if sin_theta > 0.0001:
        theta = 2 * np.arcsin(sin_theta)
        theta *= 1 if quat[0] >= 0 else -1
        axis = quat[1:] / sin_theta

    return axis, theta
