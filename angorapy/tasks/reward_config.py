"""Configuration dictionaries providing parameter collections for reard functions."""


def resolve_config_name(config_name: str):
    """Convert a name of a reward configuration into the corresponding dictionary."""
    parts = config_name.split(".")

    try:
        collection = globals()[parts[0]]
        return collection[parts[1]]
    except KeyError:
        raise KeyError("Reward configuration name could not be resolved.")
