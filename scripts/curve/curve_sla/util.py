import numpy as np


DEPTH_MAX = 2.5

def convert_depth_to_greyscale_rgb(depth_image: np.ndarray) -> np.ndarray | None:
    """
    Converts a depth image (2D or 3D with 1 channel) into a 3D greyscale RGB image.

    The depth values are normalized to the 0-255 range based on 0 to DEPTH_MAX.
    NaN and Inf values are handled.

    Args:
        depth_image (np.ndarray): A NumPy array of shape (H, W) or (H, W, 1) and
                                  dtype np.float32, representing depth.

    Returns:
        np.ndarray | None: A 3D NumPy array of shape (H, W, 3) and
                           dtype np.uint8, representing a greyscale image.
                           Returns None if input is invalid.
    """

    # --- Input Validation ---
    if not isinstance(depth_image, np.ndarray):
        print(f"Error: Input is not a NumPy array, got {type(depth_image)}.")
        return None

    # Handle (H, W, 1) by squeezing to (H, W)
    if depth_image.ndim == 3 and depth_image.shape[-1] == 1:
        depth_image = depth_image.squeeze(-1)

    # Now strictly enforce 2D
    if depth_image.ndim != 2:
        print(f"Error: Input array must be 2D (H, W) or 3D (H, W, 1), but got shape {depth_image.shape}.")
        return None

    if depth_image.size == 0:
        print("Error: Input array is empty.")
        return None

    # --- Normalization ---
    # Base on absolute values
    min_val = 0.
    max_val = DEPTH_MAX

    # Copy the image to avoid modifying the original
    normalized_image = depth_image.copy()

    # Set invalid (NaN, Inf) pixels to the min value (will become black)
    invalid_mask = ~np.isfinite(normalized_image)
    normalized_image[invalid_mask] = min_val

    # Clamp to Max - CK
    normalized_image[normalized_image > max_val] = max_val

    # Normalize the array to the range [0.0, 1.0]
    # 1 - (image - min) / (max - min)
    normalized_image = 1. - (normalized_image - min_val) / (max_val - min_val)

    # --- Scaling and Type Conversion ---
    # Scale to [0, 255] and convert to 8-bit integer
    scaled_image_8bit = (normalized_image * 255).astype(np.uint8)

    # --- Convert to 3-Channel Greyscale ---
    # Stack the 2D greyscale image 3 times along a new last axis (axis=2)
    # This makes R, G, and B channels identical
    greyscale_rgb_image = np.stack((scaled_image_8bit,) * 3, axis=-1)

    return greyscale_rgb_image