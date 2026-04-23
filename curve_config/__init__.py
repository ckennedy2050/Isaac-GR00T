#!/usr/bin/env python3

from enum import Enum

class ArmModality(Enum):
    JOINT = "joint"
    POSE = "pose"
    EEF_9D = "eef_9d"  # xyz+rot6d
    DELTA_JOINT = "djoint"
    DELTA_POSE = "dpose"