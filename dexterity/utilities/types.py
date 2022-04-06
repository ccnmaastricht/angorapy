from typing import Tuple, Union

import numpy

from common.senses import Sensation

StepTuple = Tuple[Sensation, Union[float, numpy.ndarray], bool, dict]