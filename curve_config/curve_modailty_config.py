from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.types import ModalityConfig, ActionConfig, ActionFormat, ActionRepresentation, ActionType
from gr00t.data.embodiment_tags import EmbodimentTag

from curve_config import ArmModality


########################################################################################################################
ACTION_MODALITY = ArmModality.EEF_9D

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

observation_indices = [0]
action_indices = list(range(16))

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

    action_modality = ModalityConfig(
        delta_indices=action_indices,
        modality_keys=action_keys,
    )

    state_modality = ModalityConfig(
        delta_indices=observation_indices,
        modality_keys=[
            "x",
            "y",
            "z",
            "roll",
            "pitch",
            "yaw",
            "joints",
            "gripper",
        ])

elif ACTION_MODALITY == ArmModality.EEF_9D:

    action_modality = ModalityConfig(
        delta_indices=list(range(40)),
        modality_keys=["eef_9d", "gripper", "terminate"],
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.EEF,
                format=ActionFormat.XYZ_ROT6D,
                state_key="eef_9d",
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
                state_key="gripper",
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
                state_key="terminate",
            ),
        ],
    )


    state_modality = ModalityConfig(
        delta_indices=[0],
        modality_keys=["eef_9d", "gripper", "pad"],
    )


elif ACTION_MODALITY == ArmModality.JOINT:
    action_keys = [
        "joints",
        "gripper",
        "terminate",
    ]

    action_modality = ModalityConfig(
        delta_indices=action_indices,
        modality_keys=action_keys,
    )

    state_modality = ModalityConfig(
        delta_indices=observation_indices,
        modality_keys=[
            "x",
            "y",
            "z",
            "roll",
            "pitch",
            "yaw",
            "joints",
            "gripper",
        ])

else:
    raise NotImplementedError


#state_modality_keys = [k.split(".")[-1] for k in state_keys]
#action_modality_keys = [k.split(".")[-1] for k in action_keys]

language_keys = ["annotation.human.action.task_description"]


def get_modality_config():
    video_modality = ModalityConfig(
        delta_indices=observation_indices,
        modality_keys=video_keys,
    )
    depth_modality = ModalityConfig(
        delta_indices=observation_indices,
        modality_keys=depth_keys,
    )
    # state_modality = ModalityConfig(
    #     delta_indices=observation_indices,
    #     modality_keys=state_keys,
    # )
    # action_modality = ModalityConfig(
    #     delta_indices=action_indices,
    #     modality_keys=action_keys,
    # )
    language_modality = ModalityConfig(
        delta_indices=observation_indices,
        modality_keys=language_keys,
    )
    # TODO: Apply ActionConfigs to new modalities (see gr00t/configs/data/embodiment_configs.py)
    #action_configs=action_configs

    modality_configs = {
        "video": video_modality,
        "depth": depth_modality,
        "state": state_modality,
        "action": action_modality,
        "language": language_modality,
    }
    return modality_configs

register_modality_config(get_modality_config(), embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
