from typing import Tuple, Union

import numpy

from angorapy.common.senses import Sensation

StepTuple = Tuple[Sensation, Union[float, numpy.ndarray], bool, bool, dict]
