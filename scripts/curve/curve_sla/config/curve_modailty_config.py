from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.types import ModalityConfig
from gr00t.data.embodiment_tags import EmbodimentTag


video_keys = [
    "ego_view",
    "wrist_view",
    "shoulder_view",
    #"wrist_view_depth",
    #"shoulder_view_depth"
]

state_keys = [
    "x",
    "y",
    "z",
    "roll",
    "pitch",
    "yaw",
    "gripper",
]

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
        "state": state_modality,
        "action": action_modality,
        "language": language_modality,
    }
    return modality_configs

register_modality_config(get_modality_config(), embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
