
import copy

import cv2
import numpy as np



def recolor_arm(img_seq, color, new_color):
    """
    MimicPlay recoloring adapted to the human dataset.

    Input:
        img_seq: RGB images (T, H, W, 3)

    ROI:
        x = [60, 150)
        Full image height

    Expected image size:
        180 x 100
    """

    new_img_seq = copy.deepcopy(img_seq)

    mask_img = new_img_seq <= color
    mask_img = np.sum(mask_img, axis=3) == 3

    # Person ROI verified after human crop and resize
    mask_img[:, :, :60] = False
    mask_img[:, :, 150:] = False

    # Apply 3x3 erosion independently to each frame
    kernel = np.ones((3, 3), np.uint8)

    mask_img = np.stack([
        cv2.erode(
            mask.astype(np.uint8),
            kernel
        ).astype(np.bool_)
        for mask in mask_img
    ], axis=0)

    new_img_seq[mask_img, :] = new_color

    return new_img_seq


def random_recolor_arm(img_seq, perform_aug=True, aug_p=0.5):
    """
    Apply MimicPlay arm recoloring with the original probability.
    """

    aug = np.random.choice(
        [1, 0],
        p=[aug_p, 1 - aug_p]
    )

    if aug and perform_aug:

        arm_color = [40, 40, 40]

        new_arm_color = np.random.randint(
            0, 255, size=3
        ).tolist()

        img_seq = recolor_arm(
            img_seq,
            color=arm_color,
            new_color=new_arm_color
        )

    return img_seq


def remove_joystick_copy_paste(
    img_seq,
    source_roi,
    target_roi,
):
    """
    Remove the joystick by copying an empty table region
    over the joystick region.

    Args:
        img_seq: RGB image sequence, shape (T, H, W, 3).
        source_roi: (x1, y1, x2, y2), empty table region.
        target_roi: (x1, y1, x2, y2), joystick region.

    Returns:
        Image sequence with the joystick region replaced.

    Notes:
        - Deterministic transformation.
        - Applied identically to every frame.
        - Source and target ROIs must have the same size.
        - Original images are not modified.
    """

    new_img_seq = img_seq.copy()

    sx1, sy1, sx2, sy2 = source_roi
    tx1, ty1, tx2, ty2 = target_roi

    if img_seq.ndim != 4 or img_seq.shape[-1] != 3:
        raise ValueError(
            "img_seq must have shape (T, H, W, 3)"
        )

    _, height, width, _ = img_seq.shape

    # Check ROI boundaries
    for x1, y1, x2, y2 in (source_roi, target_roi):
        if not (0 <= x1 < x2 <= width):
            raise ValueError("Invalid ROI x coordinates")

        if not (0 <= y1 < y2 <= height):
            raise ValueError("Invalid ROI y coordinates")

    source_width = sx2 - sx1
    source_height = sy2 - sy1

    target_width = tx2 - tx1
    target_height = ty2 - ty1

    if (
        source_width != target_width
        or source_height != target_height
    ):
        raise ValueError(
            "Source and target ROIs must have the same size"
        )

    # Copy the empty table patch from each frame
    table_patch = img_seq[
        :,
        sy1:sy2,
        sx1:sx2,
        :
    ].copy()

    # Paste it over the joystick region
    new_img_seq[
        :,
        ty1:ty2,
        tx1:tx2,
        :
    ] = table_patch

    return new_img_seq
