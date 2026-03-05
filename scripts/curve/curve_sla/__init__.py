#!/usr/bin/env python3

from enum import Enum

class ArmModality(Enum):
    JOINT = "joint"
    POSE = "pose"
    DELTA_POSE = "dpose"