from ._version import version

# Expose core classes/utilities for "from mcmot import Camera, MCMOTUtils"
from .Camera import Camera
from .MCMOTracker import MCMOTracker
from .ModelPlus import ModelPlus
from . import MCMOTUtils

__all__ = [
    "version",
    "Camera",
    "MCMOTracker",
    "ModelPlus",
    "MCMOTUtils",
]