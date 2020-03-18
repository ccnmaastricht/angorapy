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
