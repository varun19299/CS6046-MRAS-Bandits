"""
Meant to be imported as
from utils.typing_helper import *

To ease # imports for typing.
"""

__all__ = [
    "TYPE_CHECKING",
    "Any",
    "Array",
    "Dict",
    "List",
    "tupperware",
    "Tuple",
    "Union",
]


from typing import Dict, List, Any, Tuple, Union
from utils.tupperware import tupperware
from numpy import ndarray as Array
