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
    SUCCESS_BONUS=.1,

    # ZONING and WEIGHTHING for auxiliary fingers; default values guarantee a max reward of ~0.009 per step for this
    AUXILIARY_ZONE_RADIUS=0.06015,  # this is the initial distance to the forefinger
    AUXILIARY_PENALTY_MULTIPLIER=0.01
)


# REACH
reach = dict(
    default=dict(REACH_BASE)
)

sequential_reach = dict(
    default=dict(REACH_BASE),
    great_success=dict(REACH_BASE,
                       SUCCESS_MULTIPLIER=20)
)

# FREE REACH
free_reach = dict(
    default=REACH_BASE,
    force_punished=dict(REACH_BASE,
                        FORCE_MULTIPLIER=0.05),
    force_punished_light=dict(REACH_BASE,
                              FORCE_MULTIPLIER=0.001),
    narrow_target_zone=dict(REACH_BASE,
                            SUCCESS_DISTANCE=0.005,
                            AUXILIARY_DISTANCE_THRESHOLD_RATIO=50)
)

free_reach_positive_reinforcement = dict(
    default=REACH_BASE,
    force_punished=dict(REACH_BASE,
                        FORCE_MULTIPLIER=0.05),
)

sequential_free_reach = dict(
    default=REACH_BASE,
    great_success=dict(REACH_BASE,
                       SUCCESS_MULTIPLIER=20),
    gentle_success=dict(REACH_BASE,
                        SUCCESS_DISTANCE=0.025)
)

if __name__ == '__main__':
    print(resolve_config_name("free_reach.force_punished"))
