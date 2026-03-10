import h5py
import numpy as np
import pandas as pd
import os, sys
import json
import glob
import torch
import torchvision
#import torchcodec
#from torchcodec.encoders import VideoEncoder
import argparse
from tqdm import tqdm


current_dir = os.path.dirname(__file__)
module_dir = os.path.join(current_dir, '..')
sys.path.append(module_dir)

from curve_sla import ArmModality
from curve_sla.util import convert_depth_to_greyscale_rgb


# --- Configuration ---
FPS = 15
CHUNK_SIZE = 1000

# UPDATED MAPPING: Matches your reference hierarchy exactly
# "kinect" is often mapped to "shoulder" in these datasets.
CAMERA_MAPPING = {
    'observations/images/gripper_camera_rgb': 'observation.images.image_gripper',
    'observations/images/kinect_camera_rgb': 'observation.images.image_shoulder',
    'observations/images/workspace_camera_rgb': 'observation.images.image_workspace',
    'observations/images/gripper_camera_depth': 'observation.depth.image_gripper_depth',
    'observations/images/kinect_camera_depth': 'observation.depth.image_shoulder_depth',
    'observations/images/workspace_camera_depth': 'observation.depth.image_workspace_depth'
}

def setup_dirs(root):
    for subdir in ["data", "meta", "videos"]:
        os.makedirs(os.path.join(root, subdir), exist_ok=True)

def get_vector_stats(array):
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    return {
        "min": np.atleast_1d(np.min(array, axis=0)).tolist(),
        "max": np.atleast_1d(np.max(array, axis=0)).tolist(),
        "mean": np.atleast_1d(np.mean(array, axis=0)).tolist(),
        "std": np.atleast_1d(np.std(array, axis=0)).tolist(),
        "count": [int(array.shape[0])]
    }


def process_video_and_get_stats_torchcodec(images_np, output_path, fps, is_depth=False):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if is_depth:
        # Replace NaNs and Infs with 0.0 before processing
        images_np = np.nan_to_num(images_np, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Apply function to every frame in the sequence
        # Assuming convert_depth_to_greyscale_rgb is defined elsewhere
        processed_frames = [convert_depth_to_greyscale_rgb(frame) for frame in images_np]
        images_np = np.array(processed_frames)

    if images_np.dtype != np.uint8:
        images_np = images_np.astype(np.uint8)

    # --- SAFETY PADDING FIX ---
    last_frame = images_np[-1:]
    padded_images = np.concatenate([images_np, last_frame], axis=0)

    # Convert to Tensor
    tensor_thwc = torch.from_numpy(padded_images)

    # --- TORCHCODEC CHANGE ---
    # torchcodec.encoders.VideoEncoder expects (Frame, Channel, Height, Width).
    # We permute explicitly here.
    tensor_tchw = tensor_thwc.permute(0, 3, 1, 2).contiguous()

    # Create the encoder with the frames and FPS
    # Note: CRF options are handled differently in torchcodec or may rely on presets.
    # The default encoding is often sufficient, but check `encoder.to_file` docs
    # for specific quality control options if "crf: 22" is strictly required.
    encoder = VideoEncoder(frames=tensor_tchw, frame_rate=fps)
    encoder.to_file(output_path, codec="h264")

    # --- STATS ---
    # We can reuse the NCHW tensor we already created for the encoder
    tensor_tchw_float = tensor_tchw.float() / 255.0

    mean = torch.mean(tensor_tchw_float, dim=(0, 2, 3))
    std = torch.std(tensor_tchw_float, dim=(0, 2, 3))
    min_val = torch.amin(tensor_tchw_float, dim=(0, 2, 3))
    max_val = torch.amax(tensor_tchw_float, dim=(0, 2, 3))
    count = tensor_tchw_float.shape[0]

    def triple(t): return [[[float(x)]] for x in t]

    return {
        "min": triple(min_val), "max": triple(max_val),
        "mean": triple(mean), "std": triple(std),
        "count": [int(count)]
    }


def process_video_and_get_stats(images_np, output_path, fps, is_depth=False):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if is_depth:
        # Replace NaNs and Infs with 0.0 before processing
        #images_np = np.nan_to_num(images_np, nan=0.0, posinf=0.0, neginf=0.0)

        # Apply your function to every frame in the sequence
        processed_frames = [convert_depth_to_greyscale_rgb(frame) for frame in images_np]
        # Stack back into a single (T, H, W, 3) array
        images_np = np.array(processed_frames)

    if images_np.dtype != np.uint8:
        images_np = images_np.astype(np.uint8)

    # --- SAFETY PADDING FIX ---
    # We duplicate the last frame to ensure the video is slightly longer
    # than the parquet timestamps require. This prevents "index out of bounds"
    # errors if the encoder drops the last frame or rounding is off.
    last_frame = images_np[-1:]
    padded_images = np.concatenate([images_np, last_frame], axis=0)

    #print(f'images shape: {padded_images.shape}')

    tensor_thwc = torch.from_numpy(padded_images)
    torchvision.io.write_video(
        filename=output_path,
        video_array=tensor_thwc,
        fps=fps,
        video_codec="h264",
        options={"crf": "22",
                 'gop_size': '12'}
    )

    tensor_tchw = tensor_thwc.permute(0, 3, 1, 2).float() / 255.0
    mean = torch.mean(tensor_tchw, dim=(0, 2, 3))
    std = torch.std(tensor_tchw, dim=(0, 2, 3))
    min_val = torch.amin(tensor_tchw, dim=(0, 2, 3))
    max_val = torch.amax(tensor_tchw, dim=(0, 2, 3))
    count = tensor_tchw.shape[0]

    def triple(t): return [[[float(x)]] for x in t]

    return {
        "min": triple(min_val), "max": triple(max_val),
        "mean": triple(mean), "std": triple(std),
        "count": [int(count)]
    }

def compute_euler_deltas(euler_angles):
    """
    Computes the delta euler angles for a sequence.
    Assumes euler_angles is shape (N, 3) and in radians.
    """
    # 1. Compute the simple difference between frames (Result is N-1 x 3)
    diffs = np.diff(euler_angles, axis=0)

    # 2. Handle Wrapping: Map the result to the range [-pi, pi)
    # This fixes the jump when an angle crosses from pi to -pi
    diffs = (diffs + np.pi) % (2 * np.pi) - np.pi

    # 3. Pad the result to match the original size (N, 3)
    # We prepend a row of zeros because the first frame has no "previous" frame
    # (Alternatively, you could duplicate the first delta)
    deltas = np.pad(diffs, ((1, 0), (0, 0)), mode='constant', constant_values=0)

    return deltas

def main(input_dir, output_dir, action_modality):
    print(f"Saving LeRobot dataset to {output_dir} with modality {action_modality.name} and {len(CAMERA_MAPPING.keys())} cameras.\n")

    setup_dirs(output_dir)
    files = sorted(glob.glob(os.path.join(input_dir, "*.hdf5")))
    if not files:
        print(f"❌ No .hdf5 files found in {input_dir}")
        return

    print(f"🚀 Converting {len(files)} episodes to LeRobot v2.1...")

    task_map = {}
    next_task_idx = 0
    episodes_meta = []
    stats_list = []
    first_ep_shapes = {}

    for global_ep_idx, filepath in tqdm(enumerate(files), total=len(files), desc="Processing"):
        try:
            with h5py.File(filepath, 'r') as f:
                # --- Task ---
                raw_text = f['actions/text_instruction'][()]
                if isinstance(raw_text, bytes): instr_str = raw_text.decode('utf-8')
                elif isinstance(raw_text, np.ndarray):
                    val = raw_text.flatten()[0]
                    instr_str = val.decode('utf-8') if isinstance(val, bytes) else str(val)
                else: instr_str = str(raw_text)

                if instr_str not in task_map:
                    task_map[instr_str] = next_task_idx
                    next_task_idx += 1
                task_idx = task_map[instr_str]

                # TODO: Make this configurable
                # --- Observation ---
                state_raw = np.concatenate([
                    f['observations/pose_xyz'][:],
                    f['observations/pose_euler'][:],
                    f['observations/pose_joint'][:],
                    f['observations/pose_gripper'][:]
                ], axis=1).astype(np.float32)

                # Pad by one until we populate force data
                padding = np.zeros((state_raw.shape[0], 1), dtype=np.float32)
                state = np.concatenate([state_raw, padding], axis=1)

                if 'observation.state' not in first_ep_shapes:
                    first_ep_shapes['observation.state'] = state.shape[1]

                # --- Action ---
                if action_modality == ArmModality.POSE:
                    # state_raw = np.concatenate([
                    #     f['observations/pose_xyz'][:],
                    #     f['observations/pose_euler'][:],
                    #     f['observations/pose_gripper'][:]
                    # ], axis=1).astype(np.float32)
                    #
                    # # Pad to 8
                    # padding = np.zeros((state_raw.shape[0], 1), dtype=np.float32)
                    # state = np.concatenate([state_raw, padding], axis=1)

                    action = np.concatenate([
                        f['actions/command_pose_xyz'][:],
                        f['actions/command_pose_euler'][:],
                        f['actions/command_pose_gripper'][:],
                        f['actions/term'][:]
                    ], axis=1).astype(np.float32)

                elif action_modality == ArmModality.JOINT:
                    # state_raw = np.concatenate([
                    #     f['observations/pose_joint'][:],
                    #     f['observations/pose_gripper'][:]
                    # ], axis=1).astype(np.float32)
                    #
                    # # Pad to 8
                    # padding = np.zeros((state_raw.shape[0], 1), dtype=np.float32)
                    # state = np.concatenate([state_raw, padding], axis=1)

                    action = np.concatenate([
                        f['actions/command_pose_joint'][:],
                        f['actions/command_pose_gripper'][:],
                        f['actions/term'][:]
                    ], axis=1).astype(np.float32)

                elif action_modality == ArmModality.DELTA_POSE:
                    #state_deuler = compute_euler_deltas(f['observations/pose_euler'][:])
                    if 'actions/command_pose_deuler' in f:
                        action_deuler = f['actions/command_pose_deuler'][:]
                    else:
                        action_deuler = compute_euler_deltas(f['actions/command_pose_euler'][:])

                    # # Observations are absolute
                    # state_raw = np.concatenate([
                    #     f['observations/pose_xyz'][:],
                    #     f['observations/pose_euler'][:],
                    #     f['observations/pose_gripper'][:]
                    # ], axis=1).astype(np.float32)
                    #
                    # # Pad to 8
                    # padding = np.zeros((state_raw.shape[0], 1), dtype=np.float32)
                    # state = np.concatenate([state_raw, padding], axis=1)

                    # Actions are delta
                    action = np.concatenate([
                        f['actions/command_pose_dxyz'][:],
                        action_deuler,
                        f['actions/command_pose_gripper'][:],
                        f['actions/term'][:]
                    ], axis=1).astype(np.float32)

                else:
                    raise ValueError(f'Bad arm_modality: {action_modality}')


                if 'action' not in first_ep_shapes:
                    first_ep_shapes['action'] = action.shape[1]


                num_frames = state.shape[0]

                if 'observations/dt' in f:
                    dts = f['observations/dt'][:].flatten()

                    # ---------------------------------------------------------
                    # [FIX] Force the first dt to be a standard frame time.
                    # This neutralizes the "initialization lag" spike.
                    # ---------------------------------------------------------
                    dts[0] = 1.0 / FPS
                    timestamps = np.cumsum(dts).astype(np.float32)
                else:
                    print(f'No dt data found!!! Using fps={FPS}')
                    timestamps = np.arange(num_frames, dtype=np.float32) / FPS

                # --- Video & Stats ---
                chunk_id = global_ep_idx // CHUNK_SIZE
                chunk_str = f"chunk-{chunk_id:03d}"
                episode_file = f"episode_{global_ep_idx:06d}.mp4"

                current_ep_stats = {
                    "observation.state": get_vector_stats(state),
                    "action": get_vector_stats(action),
                    "timestamp": get_vector_stats(timestamps),
                    "frame_index": get_vector_stats(np.arange(num_frames)),
                    "episode_index": get_vector_stats(np.full(num_frames, global_ep_idx)),
                    "index": get_vector_stats(np.arange(global_ep_idx * 10000, global_ep_idx * 10000 + num_frames)),
                    "task_index": get_vector_stats(np.full(num_frames, task_idx))
                }

                for hdf5_key, lerobot_key in CAMERA_MAPPING.items():
                    images_np = f[hdf5_key][:]
                    is_depth = 'depth' in hdf5_key

                    if lerobot_key not in first_ep_shapes:
                        # Capture shape (T, H, W, C). If depth, we convert to RGB (3 channels).
                        h, w = images_np.shape[1], images_np.shape[2]
                        c = 3 if is_depth else (images_np.shape[3] if images_np.ndim > 3 else 1)
                        first_ep_shapes[lerobot_key] = (images_np.shape[0], h, w, c)

                    video_rel_dir = os.path.join("videos", chunk_str, lerobot_key)
                    video_disk_path = os.path.join(output_dir, video_rel_dir, episode_file)

                    vid_stats = process_video_and_get_stats(images_np, video_disk_path, FPS, is_depth)
                    current_ep_stats[lerobot_key] = vid_stats

                # --- Parquet (Clean) ---
                data_dict = {
                    "observation.state": list(state),
                    "action": list(action),
                    "episode_index": [global_ep_idx] * num_frames,
                    "frame_index": range(num_frames),
                    "timestamp": timestamps,
                    "next.done": [False] * (num_frames - 1) + [True],
                    "index": np.arange(global_ep_idx * 10000, global_ep_idx * 10000 + num_frames),
                    "task_index": [task_idx] * num_frames
                }

                df = pd.DataFrame(data_dict)
                parquet_dir = os.path.join(output_dir, "data", chunk_str)
                os.makedirs(parquet_dir, exist_ok=True)
                parquet_path = os.path.join(parquet_dir, f"episode_{global_ep_idx:06d}.parquet")
                df.to_parquet(parquet_path, engine='pyarrow')

                episodes_meta.append({"episode_index": global_ep_idx, "tasks": [task_idx], "length": num_frames})
                stats_list.append({"episode_index": global_ep_idx, "stats": current_ep_stats})

        except Exception as e:
            print(f"❌ Error on {filepath}: {e}")

    # --- Write Meta ---
    print("💾 Writing Metadata...")
    stats_list.sort(key=lambda x: x["episode_index"])
    with open(os.path.join(output_dir, "meta/episodes_stats.jsonl"), 'w') as f:
        for entry in stats_list: f.write(json.dumps(entry) + "\n")

    with open(os.path.join(output_dir, "meta/episodes.jsonl"), 'w') as f:
        for ep in episodes_meta: f.write(json.dumps(ep) + "\n")

    sorted_tasks = sorted(task_map.items(), key=lambda x: x[1])
    with open(os.path.join(output_dir, "meta/tasks.jsonl"), 'w') as f:
        for t_str, t_id in sorted_tasks: f.write(json.dumps({"task_index": t_id, "task": t_str}) + "\n")

    # TODO: Make configurable
    obs_names = {"motors": ["x", "y", "z", "roll", "pitch", "yaw", "joints", "gripper", "pad"]}

    if action_modality == ArmModality.POSE:
        #obs_names = {"motors": ["x", "y", "z", "roll", "pitch", "yaw", "gripper", "pad"]}
        action_names = {"motors": ["x", "y", "z", "roll", "pitch", "yaw", "gripper", "terminate"]}
    elif action_modality == ArmModality.JOINT:
        #obs_names = {"motors": ["joints", "gripper", "pad"]}
        action_names = {"motors": ["joints", "gripper", "terminate"]}
    elif action_modality == ArmModality.DELTA_POSE:
        #obs_names = {"motors": ["x", "y", "z", "roll", "pitch", "yaw", "gripper", "pad"]}
        action_names = {"motors": ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper", "terminate"]}
    else:
        raise ValueError(f'Bad arm_modality: {action_modality}')

    info = {
        "codebase_version": "v2.1",
        "robot_type": "unknown",
        "total_episodes": len(episodes_meta),
        "total_frames": sum(e['length'] for e in episodes_meta),
        "total_tasks": len(task_map),
        "total_videos": len(episodes_meta) * len(CAMERA_MAPPING),
        "total_chunks": 1,
        "chunks_size": CHUNK_SIZE,
        "fps": FPS,
        "splits": {"train": f"0:{len(files)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.state": {
                "dtype": "float32", "shape": first_ep_shapes['observation.state'],
                "names": obs_names
            },
            "action": {
                "dtype": "float32", "shape": first_ep_shapes['action'],
                "names": action_names
            },
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
            "next.done": {"dtype": "bool", "shape": [1], "names": None}
        }
    }

    for key, shape_tup in first_ep_shapes.items():
        if key in CAMERA_MAPPING.values():
            is_depth = 'depth' in key
            h, w, c = shape_tup[1], shape_tup[2], shape_tup[3]
            info["features"][key] = {
                "dtype": "video",
                "shape": [h, w, c],
                "names": ["height", "width", "depth" if is_depth else "rgb"],
                "info": {
                    "video.height": h,
                    "video.width": w,
                    "video.fps": FPS,
                    "video.codec": "h264",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": is_depth,
                    "video.channels": c,
                    "has_audio": False
                }
            }

    with open(os.path.join(output_dir, "meta/info.json"), 'w') as f:
        json.dump(info, f, indent=4)

    try:
        # Create the symbolic link
        #base_modality_path = os.path.join(output_dir, "../modality.json")
        os.symlink("../../modality.json", os.path.join(output_dir, "meta/modality.json"))
        print(f"Symbolic link to ../../modality.json created in meta/. Ensure this target file exists.")
    except FileExistsError:
        print(f"Error: A file or directory already exists at {os.path.join(output_dir, 'meta/modality.json')}.")
    except OSError as e:
        print(f"Error creating symbolic link: {e}")


    print("\n✅ Done!")
    print(f"Obs state: {obs_names}\nAction: {action_names}")


def get_lerobot_datadir(output_dir_prefix, action_modality_str, set_type):
    return os.path.join(output_dir_prefix, action_modality_str, set_type)


if __name__ == "__main__":
    # Will save to {output_dir_prefix}_{arm_modality}/{set}/
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", help="HDF5 folder")
    parser.add_argument("output_dir_prefix", help="Path + name prefix (like 'lerobot_datasets/counter_v4a')")
    parser.add_argument("action_modality", help=str([x.value for x in ArmModality]))
    parser.add_argument('--set', type=str, default="train", help="train, eval, test")
    args = parser.parse_args()

    output_dir = get_lerobot_datadir(args.output_dir_prefix, args.action_modality, args.set)
    main(args.input_dir, output_dir, ArmModality(args.action_modality))
