"""Unit tests for :mod:`angorapy.utilities.math.quaternions` — rotation math.

Quaternions encode object/goal orientations in the dexterous-manipulation tasks
(e.g. the cube-reorientation reward). Errors in this math are uniquely insidious:
a sign flip or a missing half-angle does not crash and is essentially invisible
in training curves, while quietly corrupting the orientation signal the agent
learns from.

The tests are organized around the algebraic *laws* a correct implementation
must satisfy, which are far more robust than checking arbitrary numeric outputs:

* the identity element is neutral under multiplication;
* conjugation negates only the vector part;
* a unit quaternion times its conjugate is the identity (conjugate == inverse);
* the (angle, axis) <-> quaternion encoding round-trips;
* composing the rotation on a vector matches a known geometric ground truth.

Quaternions use the scalar-first ``[w, x, y, z]`` convention throughout.
"""
import numpy as np

from angorapy.utilities.math import quaternions as q


def test_multiply_identity_is_neutral():
    """The identity quaternion ``[1, 0, 0, 0]`` is neutral under multiplication.

    Property
        ``identity * p == p`` and ``p * identity == p`` (checked on both sides
        since quaternion multiplication is non-commutative).

    Rationale
        The identity law is the cheapest, broadest catch-all for a broken
        Hamilton product (wrong term signs or component ordering).
    """
    identity = np.array([1.0, 0.0, 0.0, 0.0])
    some = np.array([0.5, 0.5, 0.5, 0.5])
    assert np.allclose(q.multiply(identity, some), some)
    assert np.allclose(q.multiply(some, identity), some)


def test_conjugate_negates_vector_part_only():
    """Conjugation flips the imaginary (vector) part and keeps the real (scalar) part.

    Property
        ``conj([w, x, y, z]) == [w, -x, -y, -z]``.

    Rationale
        The conjugate is one in-place sign flip that is easy to apply to the
        wrong component (e.g. negating ``w`` instead); this pins it exactly.
    """
    quat = np.array([0.2, 0.3, -0.4, 0.5])
    assert np.allclose(q.conjugate(quat), [0.2, -0.3, 0.4, -0.5])


def test_unit_quaternion_times_its_conjugate_is_identity():
    """For a unit quaternion, ``q * conj(q) == identity`` (the conjugate is the inverse).

    Property
        Any unit quaternion composed with its conjugate yields the identity
        rotation ``[1, 0, 0, 0]``.

    Rationale
        This inverse property underlies all rotation composition (``q v q^-1``);
        if it fails, every rotation built on conjugation is wrong.
    """
    unit = q.from_angle_and_axis(np.pi / 3, np.array([0.0, 0.0, 1.0]))
    product = q.multiply(unit, q.conjugate(unit))
    assert np.allclose(product, [1.0, 0.0, 0.0, 0.0], atol=1e-7)


def test_angle_axis_round_trip():
    """Encoding a rotation from ``(angle, axis)`` and decoding it recovers both.

    Property
        ``to_axis_angle(from_angle_and_axis(theta, axis)) == (axis, theta)`` for
        ``theta`` in ``(0, pi)`` and a unit ``axis``.

    Rationale
        Round-tripping catches the half-angle (``theta / 2``) and normalization
        details that a mere unit-norm check on the quaternion would miss.
    """
    angle = np.pi / 3
    axis = np.array([0.0, 1.0, 0.0])
    quat = q.from_angle_and_axis(angle, axis.copy())  # copy: from_angle_and_axis normalizes in place

    recovered_axis, recovered_angle = q.to_axis_angle(quat)

    assert np.isclose(recovered_angle, angle, atol=1e-6)
    assert np.allclose(recovered_axis, axis, atol=1e-6)


def test_rotate_vector_ninety_degrees_about_z():
    """Rotating the x-axis by +90 degrees about z maps it onto the y-axis.

    Property
        With the vector ``(1, 0, 0)`` encoded as the pure quaternion
        ``[0, 1, 0, 0]``, applying a +90 deg rotation about ``z`` yields the
        vector ``(0, 1, 0)`` (quaternion ``[0, 0, 1, 0]``).

    Rationale
        An end-to-end geometric ground truth that validates the full
        ``rotate`` composition (``q v conj(q)``), not just the individual
        building blocks tested above.
    """
    rotation = q.from_angle_and_axis(np.pi / 2, np.array([0.0, 0.0, 1.0]))
    x_axis = np.array([0.0, 1.0, 0.0, 0.0])  # pure quaternion encoding the vector (1,0,0)

    rotated = q.rotate(x_axis, rotation)

    # expect the vector (0,1,0) -> quaternion [0,0,1,0]
    assert np.allclose(rotated, [0.0, 0.0, 1.0, 0.0], atol=1e-7)
