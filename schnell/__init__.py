from .mapping import MapCalculator  # noqa
from .correlation import (  # noqa
    NoiseCorrelationBase,
    NoiseCorrelationConstant,
    NoiseCorrelationConstantIdentity,
    NoiseCorrelationConstantR,
    NoiseCorrelationFromFunctions,
    NoiseCorrelationLISA)
from .detector import (  # noqa
    Detector, GroundDetectorTriangle,
    GroundDetector, LISADetector, ALIADetector,
    BBOStarDetector, BBODetector)
from .space_detector import (  # noqa
    LISAlikeDetector, LISADetector2, ALIADetector2)
__version__ = '0.2.0'
