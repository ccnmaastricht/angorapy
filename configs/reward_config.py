"""Configuration dictionaries providing parameter collections for reard functions."""


def resolve_config_name(config_name: str):
    """Convert a name of a reward configuration into the corresponding dictionary."""
    parts = config_name.split(".")

    try:
        collection = globals()[parts[0]]
        return collection[parts[1]]
    except KeyError:
        raise KeyError("Reward configuration name could not be resolved.")


# BASE VALUES
REACH_BASE = dict(
    FORCE_MULTIPLIER=.0,  # scales punishment for force application
    SUCCESS_DISTANCE=.02,  # fingertip distance at which reach is a success
    SUCCESS_MULTIPLIER=.1,

    # ratio between the size of penalty zone of auxiliary fingertips and success distance of target fingertip
    AUXILIARY_DISTANCE_THRESHOLD_RATIO=10
)

# FREE REACH
free_reach = dict(
    force_punished=dict(REACH_BASE,
                        FORCE_MULTIPLIER=0.05)
)

if __name__ == '__main__':
    print(resolve_config_name("free_reach.force_punished"))
