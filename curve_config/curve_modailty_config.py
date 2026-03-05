from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.types import ModalityConfig
from gr00t.data.embodiment_tags import EmbodimentTag

import os
import sys

current_dir = os.path.dirname(__file__)
module_dir = os.path.join(current_dir, '..', 'scripts', 'curve')
sys.path.append(module_dir)

from curve_sla import ArmModality


########################################################################################################################
ACTION_MODALITY = ArmModality.JOINT

video_keys = [
    "image_workspace",
    "image_gripper",
    "image_shoulder",
]

depth_keys = [
    #"image_workspace_depth",
    "image_gripper_depth",
    "image_shoulder_depth",
]

state_keys = [
    "x",
    "y",
    "z",
    "roll",
    "pitch",
    "yaw",
    "joints",
    "gripper",
]

if ACTION_MODALITY == ArmModality.POSE:
    action_keys = [
        "x",
        "y",
        "z",
        "roll",
        "pitch",
        "yaw",
        "gripper",
        "terminate",
    ]
elif ACTION_MODALITY == ArmModality.JOINT:
    action_keys = [
        "joints",
        "gripper",
        "terminate",
    ]
else:
    raise NotImplementedError


state_modality_keys = [k.split(".")[-1] for k in state_keys]
action_modality_keys = [k.split(".")[-1] for k in action_keys]

language_keys = ["annotation.human.action.task_description"]
observation_indices = [0]
action_indices = list(range(16))

def get_modality_config():
    video_modality = ModalityConfig(
        delta_indices=observation_indices,
        modality_keys=video_keys,
    )
    depth_modality = ModalityConfig(
        delta_indices=observation_indices,
        modality_keys=depth_keys,
    )
    state_modality = ModalityConfig(
        delta_indices=observation_indices,
        modality_keys=state_keys,
    )
    action_modality = ModalityConfig(
        delta_indices=action_indices,
        modality_keys=action_keys,
    )
    language_modality = ModalityConfig(
        delta_indices=observation_indices,
        modality_keys=language_keys,
    )
    modality_configs = {
        "video": video_modality,
        "depth": depth_modality,
        "state": state_modality,
        "action": action_modality,
        "language": language_modality,
    }
    return modality_configs

register_modality_config(get_modality_config(), embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
