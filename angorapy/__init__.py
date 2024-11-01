import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ["TF_USE_LEGACY_KERAS"]= '1'

from angorapy import agent
from angorapy import common
from angorapy import models
from angorapy import tasks
from angorapy import utilities
from angorapy import analysis
from angorapy.common import policies
from angorapy.utilities.monitoring import Monitor

from angorapy.tasks.registration import make_task
from angorapy.tasks.registration import make_task as make_env  # For backwards compatibility

from angorapy.utilities import evaluation

from angorapy.agent import Agent

# Builtin models
from angorapy.models import simple
from angorapy.models import shadow
from angorapy.models import shadow_v2
