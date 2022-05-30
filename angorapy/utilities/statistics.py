import numpy as np
from numpy import ndarray as arr


def increment_mean_var(old_mean: arr, old_var: arr, new_mean: arr, new_var: arr, n: int, other_n: int = 1):
    """Incremental update of mean and variance.

    Variance based on parallel algorithm at https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance.
    """
    next_n = (n + other_n)
    mean = (n / next_n) * old_mean + (other_n / next_n) * new_mean

    delta = new_mean - old_mean
    m_a = old_var * (n - 1)
    m_b = new_var * (other_n - 1)
    m2 = m_a + m_b + delta ** 2 * n * other_n / (n + other_n)
    variance = m2 / (n + other_n - 1)

    return mean, variance


def ignore_none(func, sequence):
    """Apply function to a sequence ignoring all its Nones."""
    no_nones = [e for e in sequence if e is not None]
    if no_nones:
        return func(no_nones)

    return None


def mean_fill_nones(sequence, dtype=float):
    """In a sequence of numbers, replace Nones by the non-None boundary mean."""

    sequence = np.array(sequence, dtype=dtype)

    while np.any(np.isnan(sequence)):
        left = (np.isnan(sequence)).nonzero()[0] - 1
        right = (np.isnan(sequence)).nonzero()[0] + 1

        while np.any(np.isnan(sequence[left])):
            left[np.isnan(sequence[left])] -= 1

        while np.any(np.isnan(sequence[right])):
            right[np.isnan(sequence[right])] += 1

        sequence[np.isnan(sequence)] = (sequence[right] + sequence[left]) / 2

    return sequence


if __name__ == '__main__':
    means = [1, 2, 3, None, None, 6, 7, None, 9]
    print(mean_fill_nones(means))
