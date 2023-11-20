REACH_BASE = dict(
    FORCE_MULTIPLIER=.0,  # scales punishment for force application
    SUCCESS_DISTANCE=.02,  # fingertip distance at which reach is a success
    SUCCESS_BONUS=.1,

    # ZONING and WEIGHTHING for auxiliary fingers; default values guarantee a max reward of ~0.009 per step for this
    AUXILIARY_ZONE_RADIUS=0.06015,  # this is the initial distance to the forefinger
    AUXILIARY_PENALTY_MULTIPLIER=0.01
)
MANIPULATE_BASE = dict(
    SUCCESS_BONUS=5,
    DROPPING_PENALTY=20,
    FORCE_MULTIPLIER=.0,
    TENDON_STRESS_MULTIPLIER=.0,
)
reach = dict(
    default=dict(REACH_BASE),
    force_punished=dict(REACH_BASE, FORCE_MULTIPLIER=0.05),
)
sequential_reach = dict(
    default=dict(REACH_BASE,
                 SUCCESS_DISTANCE=.5),
    great_success=dict(REACH_BASE,
                       SUCCESS_MULTIPLIER=5)
)
free_reach = dict(
    default=REACH_BASE,
    force_punished=dict(REACH_BASE, FORCE_MULTIPLIER=0.05),
    force_punished_light=dict(REACH_BASE, FORCE_MULTIPLIER=0.001),
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
manipulate = dict(
    default=MANIPULATE_BASE,
    penalized_force=dict(MANIPULATE_BASE,
                         FORCE_MULTIPLIER=0.001, ),
    penalized_anatomy=dict(MANIPULATE_BASE,
                           TENDON_STRESS_MULTIPLIER=0.01,
                           FORCE_MULTIPLIER=0.01, )
)
