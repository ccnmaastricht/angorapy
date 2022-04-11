from typing import Tuple, Union

import numpy

from dexterity.common.senses import Sensation

StepTuple = Tuple[Sensation, Union[float, numpy.ndarray], bool, dict]
