
import argparse
from pathlib import Path

import cv2
import numpy as np

from hem.datasets.util import crop, resize
from mtlfd_adaptation.trajectory_bridge import load_traj
from mtlfd_adaptation.augmentations import (
    remove_joystick_copy_paste,
    recolor_arm,
)


def save_rgb(path, image):
    image = np.clip(image, 0, 255).astype(np.uint8)
    cv2.imwrite(
        str(path),
        cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    )


def draw_roi(image, roi, color, label):
    result = image.copy()
    x1, y1, x2, y2 = roi

    cv2.rectangle(
        result,
        (x1, y1),
        (x2 - 1, y2 - 1),
        color,
        2
    )

    cv2.putText(
        result,
        label,
        (x1, max(15, y1 - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        cv2.LINE_AA
    )

    return result


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--pkl", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--frame", type=int, default=1)
    parser.add_argument("--is-real", action="store_true")

    # Human agent crop and resize
    parser.add_argument(
        "--crop",
        type=int,
        nargs=4,
        default=[0, 30, 140, 120]
    )
    parser.add_argument("--width", type=int, default=180)
    parser.add_argument("--height", type=int, default=100)

    # Person ROI on resized image
    parser.add_argument("--x1", type=int, default=60)
    parser.add_argument("--x2", type=int, default=150)
    parser.add_argument("--threshold", type=int, default=40)

    # Joystick ROIs on ORIGINAL image
    parser.add_argument(
        "--source-roi",
        type=int,
        nargs=4,
        default=[332, 98, 486, 136]
    )

    parser.add_argument(
        "--target-roi",
        type=int,
        nargs=4,
        default=[333, 59, 487, 97]
    )

    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # 1. LOAD ORIGINAL FRAME
    # --------------------------------------------------

    traj, _ = load_traj(
        args.pkl,
        is_real=args.is_real
    )

    if not 0 <= args.frame < len(traj):
        raise ValueError("Frame index out of range")

    original = np.asarray(
        traj.get(args.frame)["obs"]["image"]
    ).copy()

    print("Original image:", original.shape)

    save_rgb(output / "01_original.png", original)

    # --------------------------------------------------
    # 2. JOYSTICK ROI VISUALIZATION
    # --------------------------------------------------

    source_roi = tuple(args.source_roi)
    target_roi = tuple(args.target_roi)

    joystick_overlay = draw_roi(
        original,
        source_roi,
        (0, 255, 0),
        "SOURCE TABLE"
    )

    joystick_overlay = draw_roi(
        joystick_overlay,
        target_roi,
        (255, 0, 0),
        "TARGET JOYSTICK"
    )

    save_rgb(
        output / "02_joystick_rois.png",
        joystick_overlay
    )

    # --------------------------------------------------
    # 3. JOYSTICK COPY-PASTE
    # --------------------------------------------------

    joystick_removed = remove_joystick_copy_paste(
        original[None],
        source_roi=source_roi,
        target_roi=target_roi
    )[0]

    save_rgb(
        output / "03_joystick_removed_original.png",
        joystick_removed
    )

    # --------------------------------------------------
    # 4. CROP AND RESIZE
    # --------------------------------------------------

    cropped_original = crop(
        original,
        args.crop
    )

    cropped_modified = crop(
        joystick_removed,
        args.crop
    )

    resized_original = resize(
        cropped_original,
        (args.width, args.height)
    ).astype(np.uint8)

    resized_modified = resize(
        cropped_modified,
        (args.width, args.height)
    ).astype(np.uint8)

    save_rgb(
        output / "04_resized_original.png",
        resized_original
    )

    save_rgb(
        output / "05_resized_joystick_removed.png",
        resized_modified
    )

    # --------------------------------------------------
    # 5. PERSON ROI VISUALIZATION
    # --------------------------------------------------

    height, width = resized_modified.shape[:2]

    if not 0 <= args.x1 < args.x2 <= width:
        raise ValueError("Invalid person ROI")

    person_overlay = draw_roi(
        resized_modified,
        (args.x1, 0, args.x2, height),
        (255, 0, 0),
        "PERSON ROI"
    )

    save_rgb(
        output / "06_person_roi.png",
        person_overlay
    )

    # --------------------------------------------------
    # 6. PERSON RECOLORING PREVIEW
    # --------------------------------------------------

    # Keep the mask calculation explicit so that --x1,
    # --x2 and --threshold can be tested from CLI.
    mask = np.all(
        resized_modified <= args.threshold,
        axis=2
    )

    mask[:, :args.x1] = False
    mask[:, args.x2:] = False

    kernel = np.ones((3, 3), np.uint8)

    mask = cv2.erode(
        mask.astype(np.uint8),
        kernel
    ).astype(bool)

    cv2.imwrite(
        str(output / "07_person_mask.png"),
        mask.astype(np.uint8) * 255
    )

    # Preview with a fixed RGB color
    recolored = resized_modified.copy()
    recolored[mask] = [200, 60, 200]

    save_rgb(
        output / "08_person_recolored.png",
        recolored
    )

    # --------------------------------------------------
    # 7. FINAL COMPARISON
    # --------------------------------------------------

    comparison = np.concatenate(
        [
            resized_original,
            resized_modified,
            recolored
        ],
        axis=1
    )

    comparison = cv2.resize(
        comparison,
        None,
        fx=3,
        fy=3,
        interpolation=cv2.INTER_NEAREST
    )

    save_rgb(
        output / "09_final_comparison.png",
        comparison
    )

    print("\n--- AUGMENTATION TEST ---")
    print("Original shape:", original.shape)
    print("Crop:", args.crop)
    print("Final resolution:", resized_modified.shape)
    print("Joystick SOURCE:", source_roi)
    print("Joystick TARGET:", target_roi)
    print("Person ROI:", (args.x1, 0, args.x2, height))
    print("Person selected pixels:", int(mask.sum()))
    print("Output:", output)


if __name__ == "__main__":
    main()
