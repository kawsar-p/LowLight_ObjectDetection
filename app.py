import cv2
import numpy as np
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO


# ============================================================
# SETTINGS
# ============================================================

INPUT_DIR = Path("test_images") / "new test"

BASE_OUTPUT_DIR = (
    Path("test_images")
    / "test result"
)

# Every execution gets a unique timestamp-based folder.
RUN_TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

OUTPUT_DIR = BASE_OUTPUT_DIR / f"run_{RUN_TIMESTAMP}"

ORIGINAL_OUTPUT = OUTPUT_DIR / "original"
ENHANCED_OUTPUT = OUTPUT_DIR / "enhanced"
FINAL_OUTPUT = OUTPUT_DIR / "final"

MODEL_PATH = Path("FINAL_MODEL") / "best.pt"

VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}

# YOLO candidate threshold
CONFIDENCE = 0.10

# Final confident detection threshold
UNKNOWN_THRESHOLD = 0.50

# Enhanced result must improve enough to replace original
MIN_IMPROVEMENT = 0.04

IMAGE_SIZE = 640


# ============================================================
# IMAGE ANALYSIS
# ============================================================

def analyze_image(img):

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    (
        p5,
        p10,
        p25,
        median,
        p75,
        p90,
        p95
    ) = np.percentile(
        gray,
        [5, 10, 25, 50, 75, 90, 95]
    )

    brightness = float(
        np.mean(gray)
    )

    contrast = float(
        np.std(gray)
    )

    dark_ratio = float(
        np.mean(gray < 50)
    )

    very_dark_ratio = float(
        np.mean(gray < 30)
    )

    shadow_ratio = float(
        np.mean(gray < 75)
    )

    highlight_ratio = float(
        np.mean(gray > 245)
    )

    lap = cv2.Laplacian(
        gray,
        cv2.CV_64F
    )

    sharpness = float(
        lap.var()
    )

    return {
        "brightness": brightness,
        "median": float(median),
        "p5": float(p5),
        "p10": float(p10),
        "p25": float(p25),
        "p75": float(p75),
        "p90": float(p90),
        "p95": float(p95),
        "contrast": contrast,
        "dark_ratio": dark_ratio,
        "very_dark_ratio": very_dark_ratio,
        "shadow_ratio": shadow_ratio,
        "highlight_ratio": highlight_ratio,
        "sharpness": sharpness
    }


# ============================================================
# GAMMA CORRECTION
# ============================================================

def apply_gamma(img, gamma):

    if gamma >= 0.995:
        return img.copy()

    table = np.array(
        [
            np.clip(
                ((i / 255.0) ** gamma) * 255.0,
                0,
                255
            )
            for i in range(256)
        ],
        dtype=np.uint8
    )

    return cv2.LUT(
        img,
        table
    )


# ============================================================
# LOCAL CONTRAST ENHANCEMENT
# ============================================================

def apply_clahe(
    img,
    clip_limit
):

    lab = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2LAB
    )

    l, a, b = cv2.split(
        lab
    )

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(8, 8)
    )

    enhanced_l = clahe.apply(
        l
    )

    result = cv2.cvtColor(
        cv2.merge(
            (
                enhanced_l,
                a,
                b
            )
        ),
        cv2.COLOR_LAB2BGR
    )

    return result


# ============================================================
# SAFE DENOISING
# ============================================================

def apply_denoising(
    img,
    metrics
):

    median = metrics["median"]
    contrast = metrics["contrast"]
    sharpness = metrics["sharpness"]

    # High sharpness alone is NOT considered noise.
    #
    # Denoise only when the image is dark,
    # low contrast and has suspicious high-frequency noise.

    if (
        median < 75
        and contrast < 30
        and sharpness > 500
    ):

        return cv2.fastNlMeansDenoisingColored(
            img,
            None,
            2,
            2,
            7,
            21
        )

    return img


# ============================================================
# DETAIL PRESERVATION
# ============================================================

def preserve_details(
    img,
    metrics
):

    contrast = metrics["contrast"]

    if contrast < 22:

        amount = 0.10

    elif contrast < 35:

        amount = 0.07

    else:

        amount = 0.04

    blur = cv2.GaussianBlur(
        img,
        (0, 0),
        1.0
    )

    result = cv2.addWeighted(
        img,
        1.0 + amount,
        blur,
        -amount,
        0
    )

    return result


# ============================================================
# SAFE BLEND
# ============================================================

def blend_images(
    original,
    enhanced,
    strength
):

    return cv2.addWeighted(
        original,
        1.0 - strength,
        enhanced,
        strength,
        0
    )


# ============================================================
# ADAPTIVE ENHANCEMENT
# ============================================================

def enhance_image(img):

    metrics = analyze_image(
        img
    )

    median = metrics["median"]
    brightness = metrics["brightness"]
    contrast = metrics["contrast"]

    dark_ratio = metrics[
        "dark_ratio"
    ]

    very_dark_ratio = metrics[
        "very_dark_ratio"
    ]

    shadow_ratio = metrics[
        "shadow_ratio"
    ]

    highlight_ratio = metrics[
        "highlight_ratio"
    ]

    # ========================================================
    # CASE 1: ALREADY CLEAR
    # ========================================================

    if (
        median >= 115
        and brightness >= 115
        and contrast >= 35
        and dark_ratio < 0.15
    ):

        return (
            img.copy(),
            "ORIGINAL_CLEAR",
            metrics
        )

    # ========================================================
    # CASE 2: EXTREMELY DARK
    # ========================================================

    if (
        median < 30
        and very_dark_ratio > 0.45
    ):

        gamma = 0.45
        clip = 2.0
        strength = 0.95

        method = (
            "VERY_DARK_STRONG"
        )

    # ========================================================
    # CASE 3: VERY DARK + LOW CONTRAST
    # ========================================================

    elif (
        median < 40
        and contrast < 25
    ):

        gamma = 0.50
        clip = 1.9
        strength = 0.93

        method = (
            "VERY_DARK_LOW_CONTRAST"
        )

    # ========================================================
    # CASE 4: VERY DARK + GOOD DETAIL
    # ========================================================

    elif (
        median < 40
        and contrast >= 25
    ):

        gamma = 0.56
        clip = 1.45
        strength = 0.88

        method = (
            "VERY_DARK_DETAIL"
        )

    # ========================================================
    # CASE 5: DARK + LOW CONTRAST
    # ========================================================

    elif (
        median < 55
        and contrast < 30
    ):

        gamma = 0.61
        clip = 1.75
        strength = 0.84

        method = (
            "DARK_LOW_CONTRAST"
        )

    # ========================================================
    # CASE 6: DARK + HIGH SHADOW AREA
    # ========================================================

    elif (
        median < 60
        and shadow_ratio > 0.55
    ):

        gamma = 0.65
        clip = 1.55
        strength = 0.82

        method = (
            "DARK_SHADOW_RECOVERY"
        )

    # ========================================================
    # CASE 7: DARK + NORMAL CONTRAST
    # ========================================================

    elif (
        median < 65
        and contrast >= 30
    ):

        gamma = 0.68
        clip = 1.30
        strength = 0.77

        method = (
            "DARK_NORMAL_CONTRAST"
        )

    # ========================================================
    # CASE 8: MODERATELY DARK + LOW CONTRAST
    # ========================================================

    elif (
        median < 80
        and contrast < 30
    ):

        gamma = 0.73
        clip = 1.55
        strength = 0.70

        method = (
            "MODERATE_LOW_CONTRAST"
        )

    # ========================================================
    # CASE 9: MODERATELY DARK + NORMAL CONTRAST
    # ========================================================

    elif (
        median < 80
        and contrast >= 30
    ):

        gamma = 0.77
        clip = 1.20
        strength = 0.63

        method = (
            "MODERATE_NORMAL"
        )

    # ========================================================
    # CASE 10: SLIGHTLY DARK + LOW CONTRAST
    # ========================================================

    elif (
        median < 100
        and contrast < 32
    ):

        gamma = 0.84
        clip = 1.30
        strength = 0.53

        method = (
            "SLIGHT_DARK_LOW_CONTRAST"
        )

    # ========================================================
    # CASE 11: SLIGHTLY DARK
    # ========================================================

    elif (
        median < 115
    ):

        gamma = 0.91
        clip = 1.10
        strength = 0.40

        method = (
            "SLIGHT_DARK"
        )

    # ========================================================
    # CASE 12: OTHERWISE
    # ========================================================

    else:

        gamma = 0.96
        clip = 1.05
        strength = 0.25

        method = (
            "MINIMAL"
        )

    # ========================================================
    # GAMMA
    # ========================================================

    result = apply_gamma(
        img,
        gamma
    )

    # ========================================================
    # CLAHE
    # ========================================================

    result = apply_clahe(
        result,
        clip
    )

    # ========================================================
    # DENOISING
    # ========================================================

    result = apply_denoising(
        result,
        metrics
    )

    # ========================================================
    # DETAIL PRESERVATION
    # ========================================================

    result = preserve_details(
        result,
        metrics
    )

    # ========================================================
    # BLEND
    # ========================================================

    result = blend_images(
        img,
        result,
        strength
    )

    # ========================================================
    # HIGHLIGHT PROTECTION
    # ========================================================

    result_gray = cv2.cvtColor(
        result,
        cv2.COLOR_BGR2GRAY
    )

    result_median = float(
        np.median(result_gray)
    )

    result_highlight_ratio = float(
        np.mean(result_gray > 245)
    )

    if (
        result_median > 190
        or result_highlight_ratio > 0.08
    ):

        result = cv2.addWeighted(
            img,
            0.40,
            result,
            0.60,
            0
        )

        method += "_SAFE"

    # ========================================================
    # FINAL QUALITY CHECK
    # ========================================================

    # Avoid enhancement becoming almost identical
    # when the original is genuinely dark.

    if (
        median < 80
        and
        np.mean(
            cv2.cvtColor(
                result,
                cv2.COLOR_BGR2GRAY
            )
        )
        <= brightness + 3
    ):

        stronger = apply_gamma(
            img,
            max(
                gamma - 0.05,
                0.45
            )
        )

        stronger = apply_clahe(
            stronger,
            min(
                clip + 0.15,
                2.0
            )
        )

        result = cv2.addWeighted(
            img,
            0.20,
            stronger,
            0.80,
            0
        )

        method += "_BOOST"

    metrics["enhanced_median"] = float(
        np.median(
            cv2.cvtColor(
                result,
                cv2.COLOR_BGR2GRAY
            )
        )
    )

    return (
        result,
        method,
        metrics
    )


# ============================================================
# YOLO DETECTION
# ============================================================

def run_detection(
    model,
    image
):

    results = model.predict(
        source=image,
        conf=CONFIDENCE,
        imgsz=IMAGE_SIZE,
        save=False,
        verbose=False
    )

    return results[0]


# ============================================================
# EXTRACT CONFIDENT DETECTIONS
# ============================================================

def get_detections(
    result,
    model
):

    detections = []

    if result.boxes is None:
        return detections

    for box in result.boxes:

        confidence = float(
            box.conf[0]
        )

        if confidence < UNKNOWN_THRESHOLD:
            continue

        class_id = int(
            box.cls[0]
        )

        class_name = model.names[
            class_id
        ]

        xyxy = (
            box.xyxy[0]
            .cpu()
            .numpy()
        )

        detections.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "confidence": confidence,
                "box": xyxy
            }
        )

    return detections


# ============================================================
# DETECTION QUALITY SCORE
# ============================================================

def calculate_detection_score(
    detections
):

    if not detections:
        return 0.0

    confidences = [
        d["confidence"]
        for d in detections
    ]

    max_conf = max(
        confidences
    )

    avg_conf = (
        sum(confidences)
        /
        len(confidences)
    )

    # Small bonus for finding multiple objects.
    object_bonus = min(
        len(detections) * 0.025,
        0.10
    )

    score = (
        (max_conf * 0.55)
        +
        (avg_conf * 0.40)
        +
        object_bonus
    )

    return float(score)


# ============================================================
# DRAW FINAL RESULT
# ============================================================

def draw_final_result(
    image,
    detections,
    source
):
    """
    Draw clean, resolution-aware detection results.

    Important:
    Small images such as 128x160 must NOT receive the same
    fixed-size labels/borders as large images.
    """

    output = image.copy()

    height, width = output.shape[:2]
    min_dim = min(width, height)

    # ========================================================
    # ADAPTIVE VISUAL SCALE
    # ========================================================
    # The previous version forced scale >= 0.85.
    # That made labels/banners too large on tiny images.
    #
    # New behavior:
    #   128x160 -> small label / thin box
    #   640x640 -> medium label / normal box
    #   large image -> larger label, but still capped
    # ========================================================

    visual_scale = np.clip(
        min_dim / 320.0,
        0.35,
        1.35
    )

    box_thickness = max(
        1,
        int(round(2.2 * visual_scale))
    )

    font_scale = float(
        np.clip(
            0.42 * visual_scale,
            0.30,
            0.75
        )
    )

    text_thickness = max(
        1,
        int(round(1.5 * visual_scale))
    )

    # ========================================================
    # DRAW OBJECTS
    # ========================================================

    for detection in detections:

        x1, y1, x2, y2 = [
            int(round(v))
            for v in detection["box"]
        ]

        # Keep coordinates inside the image.
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width - 1))
        y2 = max(0, min(y2, height - 1))

        class_name = str(
            detection["class_name"]
        )

        confidence = float(
            detection["confidence"]
        )

        label = (
            f"{class_name} "
            f"{confidence * 100:.1f}%"
        )

        # ====================================================
        # BOUNDING BOX
        # ====================================================

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            box_thickness,
            cv2.LINE_AA
        )

        # ====================================================
        # LABEL SIZE
        # ====================================================

        (
            text_width,
            text_height
        ), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_thickness
        )

        padding_x = max(
            3,
            int(round(4 * visual_scale))
        )

        padding_y = max(
            2,
            int(round(3 * visual_scale))
        )

        label_width = (
            text_width
            +
            padding_x * 2
        )

        label_height = (
            text_height
            +
            baseline
            +
            padding_y * 2
        )

        # ====================================================
        # LABEL POSITION
        # ====================================================

        # Prefer above the box.
        label_x = x1
        label_y = y1 - label_height - 2

        # If there is not enough room above, put it below.
        if label_y < 0:

            label_y = y1 + 2

        # Keep the label inside the image.
        if label_x + label_width > width:

            label_x = max(
                width - label_width,
                0
            )

        if label_y + label_height > height:

            label_y = max(
                height - label_height,
                0
            )

        # ====================================================
        # LABEL BACKGROUND
        # ====================================================

        cv2.rectangle(
            output,
            (
                label_x,
                label_y
            ),
            (
                min(
                    label_x + label_width,
                    width - 1
                ),
                min(
                    label_y + label_height,
                    height - 1
                )
            ),
            (0, 255, 0),
            -1
        )

        # ====================================================
        # LABEL TEXT
        # ====================================================

        text_x = (
            label_x
            +
            padding_x
        )

        text_y = (
            label_y
            +
            padding_y
            +
            text_height
        )

        cv2.putText(
            output,
            label,
            (
                text_x,
                text_y
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            text_thickness,
            cv2.LINE_AA
        )

    # ========================================================
    # FINAL SOURCE INDICATOR
    # ========================================================
    #
    # Do not use a huge fixed 45px banner on small images.
    # For very small images use a compact top strip.
    # ========================================================

    banner_height = max(
        18,
        int(round(height * 0.10))
    )

    banner_height = min(
        banner_height,
        max(24, int(height * 0.20))
    )

    # Source text is kept short so it fits small images.
    source_text = f"FINAL: {source}"

    source_font_scale = float(
        np.clip(
            min_dim / 420.0,
            0.28,
            0.58
        )
    )

    source_thickness = max(
        1,
        int(round(
            1.5 * np.clip(
                min_dim / 320.0,
                0.50,
                1.0
            )
        ))
    )

    (
        source_text_width,
        source_text_height
    ), source_baseline = cv2.getTextSize(
        source_text,
        cv2.FONT_HERSHEY_SIMPLEX,
        source_font_scale,
        source_thickness
    )

    # If the source label is still too wide, use only the source.
    if source_text_width > width - 8:

        source_text = source

        (
            source_text_width,
            source_text_height
        ), source_baseline = cv2.getTextSize(
            source_text,
            cv2.FONT_HERSHEY_SIMPLEX,
            source_font_scale,
            source_thickness
        )

    # Last-resort compact label for extremely tiny images.
    if source_text_width > width - 8:

        source_text = "FINAL"

        (
            source_text_width,
            source_text_height
        ), source_baseline = cv2.getTextSize(
            source_text,
            cv2.FONT_HERSHEY_SIMPLEX,
            source_font_scale,
            source_thickness
        )

    cv2.rectangle(
        output,
        (0, 0),
        (
            width - 1,
            min(
                banner_height,
                height - 1
            )
        ),
        (0, 0, 0),
        -1
    )

    source_x = max(
        3,
        (width - source_text_width) // 2
    )

    source_y = min(
        height - 1,
        max(
            source_text_height + 2,
            (
                banner_height
                +
                source_text_height
                -
                source_baseline
            )
            // 2
        )
    )

    cv2.putText(
        output,
        source_text,
        (
            source_x,
            source_y
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        source_font_scale,
        (255, 255, 255),
        source_thickness,
        cv2.LINE_AA
    )

    return output


# ============================================================
# UNKNOWN RESULT
# ============================================================

def create_unknown_image(
    image
):

    output = image.copy()

    height, width = (
        output.shape[:2]
    )

    banner_height = max(
        int(height * 0.09),
        60
    )

    cv2.rectangle(
        output,
        (0, 0),
        (
            width,
            banner_height
        ),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        output,
        "UNKNOWN / NO CONFIDENT OBJECT",
        (
            15,
            int(banner_height * 0.68)
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.80,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    return output


# ============================================================
# PRINT DETECTIONS
# ============================================================

def print_detections(
    title,
    detections
):

    print(
        f"\n[{title}]"
    )

    if not detections:

        print(
            "No confident objects."
        )

        return

    for i, detection in enumerate(
        detections,
        1
    ):

        print(
            f"{i}. "
            f"{detection['class_name']} "
            f"| "
            f"{detection['confidence'] * 100:.2f}%"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # ADAPTIVE LOW-LIGHT OBJECT DETECTION PIPELINE
    # ========================================================
    #
    # SUPPORTED COMMANDS
    #
    # 1. ALL IMAGES
    #    python test_pipeline.py
    #
    # 2. ONE IMAGE BY NUMBER
    #    python test_pipeline.py 1
    #
    # 3. IMAGE RANGE (INCLUSIVE)
    #    python test_pipeline.py 1:20
    #
    # 4. MULTIPLE IMAGE NUMBERS
    #    python test_pipeline.py 1,2,3,4
    #
    # 5. MIXED NUMBERS / RANGES
    #    python test_pipeline.py 1,3,5:10,20
    #
    # 6. ONE IMAGE BY FILE NAME
    #    python test_pipeline.py image.png
    #
    # 7. MULTIPLE FILE NAMES
    #    python test_pipeline.py image1.png image2.png
    #
    # IMPORTANT:
    # Image numbers are based on the sorted file list in
    # test_images\new test.
    #
    # Ranges are INCLUSIVE:
    #   1:20 means image 1 through image 20.
    # ========================================================

    print("=" * 80)
    print("ADAPTIVE LOW-LIGHT OBJECT DETECTION")
    print("=" * 80)

    print("\n[DEBUG]")
    print(
        f"Script : "
        f"{Path(sys.argv[0]).resolve()}"
    )
    print(
        f"Args   : "
        f"{sys.argv}"
    )

    # ========================================================
    # INPUT FOLDER CHECK
    # ========================================================

    if not INPUT_DIR.exists():

        print(
            "\nERROR: Input folder not found:"
        )

        print(
            INPUT_DIR.resolve()
        )

        return

    # ========================================================
    # GET SORTED IMAGE LIST
    # ========================================================
    #
    # This same sorted list defines image numbers.
    # ========================================================

    all_images = sorted(
        [
            p
            for p in INPUT_DIR.iterdir()
            if p.is_file()
            and p.suffix.lower()
            in VALID_EXTENSIONS
        ]
    )

    if not all_images:

        print(
            "\nERROR: No valid images found in:"
        )

        print(
            INPUT_DIR.resolve()
        )

        return

    # ========================================================
    # SELECT IMAGES
    # ========================================================

    arguments = sys.argv[1:]

    # --------------------------------------------------------
    # BATCH MODE: NO ARGUMENT
    # --------------------------------------------------------

    if not arguments:

        print(
            "\n[MODE] ALL IMAGES"
        )

        image_files = all_images

    else:

        image_files = []

        print(
            "\n[MODE] SELECTED IMAGE(S)"
        )

        # ----------------------------------------------------
        # HELPER: ADD IMAGE BY 1-BASED NUMBER
        # ----------------------------------------------------

        def add_by_number(number):

            if number < 1 or number > len(all_images):

                print(
                    f"WARNING: Image number {number} "
                    f"is outside 1-{len(all_images)}. "
                    f"Skipped."
                )

                return

            selected = all_images[number - 1]

            if selected not in image_files:

                image_files.append(
                    selected
                )

        # ----------------------------------------------------
        # PROCESS EACH ARGUMENT
        # ----------------------------------------------------

        for argument in arguments:

            argument = argument.strip()

            if not argument:

                continue

            # =================================================
            # RANGE: 1:20
            # =================================================

            if ":" in argument:

                parts = argument.split(":")

                if len(parts) != 2:

                    print(
                        f"WARNING: Invalid range "
                        f"'{argument}'. Skipped."
                    )

                    continue

                start_text = parts[0].strip()
                end_text = parts[1].strip()

                try:

                    start_number = int(
                        start_text
                    )

                    end_number = int(
                        end_text
                    )

                except ValueError:

                    print(
                        f"WARNING: Invalid range "
                        f"'{argument}'. Skipped."
                    )

                    continue

                if start_number > end_number:

                    start_number, end_number = (
                        end_number,
                        start_number
                    )

                for number in range(
                    start_number,
                    end_number + 1
                ):

                    add_by_number(
                        number
                    )

                continue

            # =================================================
            # COMMA LIST: 1,2,3,4
            # =================================================

            if "," in argument:

                pieces = argument.split(",")

                for piece in pieces:

                    piece = piece.strip()

                    if not piece:

                        continue

                    # Allow comma list to contain ranges too.
                    if ":" in piece:

                        sub_parts = (
                            piece.split(":")
                        )

                        if len(sub_parts) != 2:

                            print(
                                f"WARNING: Invalid "
                                f"selection '{piece}'. "
                                f"Skipped."
                            )

                            continue

                        try:

                            sub_start = int(
                                sub_parts[0].strip()
                            )

                            sub_end = int(
                                sub_parts[1].strip()
                            )

                        except ValueError:

                            print(
                                f"WARNING: Invalid "
                                f"selection '{piece}'. "
                                f"Skipped."
                            )

                            continue

                        if sub_start > sub_end:

                            sub_start, sub_end = (
                                sub_end,
                                sub_start
                            )

                        for number in range(
                            sub_start,
                            sub_end + 1
                        ):

                            add_by_number(
                                number
                            )

                    else:

                        try:

                            add_by_number(
                                int(piece)
                            )

                        except ValueError:

                            print(
                                f"WARNING: Invalid "
                                f"image number "
                                f"'{piece}'. Skipped."
                            )

                continue

            # =================================================
            # NUMBER: 1
            # =================================================

            if argument.isdigit():

                add_by_number(
                    int(argument)
                )

                continue

            # =================================================
            # FILE NAME / PATH
            # =================================================

            requested = Path(
                argument
            )

            # Full or relative path.
            if (
                requested.exists()
                and requested.is_file()
            ):

                if requested not in image_files:

                    image_files.append(
                        requested
                    )

                continue

            # Filename inside test folder.
            requested_in_test = (
                INPUT_DIR / requested.name
            )

            if (
                requested_in_test.exists()
                and requested_in_test.is_file()
            ):

                if (
                    requested_in_test
                    not in image_files
                ):

                    image_files.append(
                        requested_in_test
                    )

                continue

            print(
                f"WARNING: Image not found "
                f"'{argument}'. Skipped."
            )

    # ========================================================
    # VALIDATE SELECTION
    # ========================================================

    if not image_files:

        print(
            "\nERROR: No images selected."
        )

        print(
            "\nAvailable image range:"
        )

        print(
            f"1-{len(all_images)}"
        )

        return

    print(
        f"\nAvailable images : "
        f"{len(all_images)}"
    )

    print(
        f"Selected images  : "
        f"{len(image_files)}"
    )

    # ========================================================
    # SHOW SELECTED IMAGES
    # ========================================================

    print(
        "\n[SELECTED FILES]"
    )

    for index, path in enumerate(
        image_files,
        1
    ):

        # Find original 1-based number where possible.
        try:

            original_number = (
                all_images.index(path) + 1
            )

        except ValueError:

            original_number = "custom"

        print(
            f"{index}. "
            f"[Image {original_number}] "
            f"{path.name}"
        )

    # ========================================================
    # CREATE UNIQUE OUTPUT FOLDER
    # ========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    ORIGINAL_OUTPUT.mkdir(
        parents=True,
        exist_ok=True
    )

    ENHANCED_OUTPUT.mkdir(
        parents=True,
        exist_ok=True
    )

    FINAL_OUTPUT.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        "\n[OUTPUT]"
    )

    print(
        OUTPUT_DIR.resolve()
    )

    print(
        "\nPipeline:"
    )

    print(
        "Original -> Adaptive Enhancement -> "
        "YOLO Original + YOLO Enhanced -> "
        "Best Result -> Final"
    )

    # ========================================================
    # CHECK MODEL
    # ========================================================

    if not MODEL_PATH.exists():

        print(
            "\nERROR: Model not found:"
        )

        print(
            MODEL_PATH.resolve()
        )

        return

    # ========================================================
    # LOAD MODEL
    # ========================================================

    print(
        "\nLoading YOLO model..."
    )

    model = YOLO(
        str(MODEL_PATH)
    )

    print(
        "Model loaded successfully."
    )

    # ========================================================
    # STATISTICS
    # ========================================================

    statistics = {
        "original": 0,
        "enhanced": 0,
        "unknown": 0
    }

    processed = 0
    failed = 0

    # ========================================================
    # PROCESS SELECTED IMAGES
    # ========================================================

    for index, image_path in enumerate(
        image_files,
        1
    ):

        print(
            "\n" + "=" * 80
        )

        print(
            f"IMAGE {index}/{len(image_files)}"
        )

        print(
            image_path.name
        )

        print(
            "=" * 80
        )

        # ====================================================
        # READ ORIGINAL
        # ====================================================

        original = cv2.imread(
            str(image_path)
        )

        if original is None:

            print(
                "ERROR: Could not read image."
            )

            failed += 1

            continue

        # ====================================================
        # SAVE ORIGINAL
        # ====================================================

        original_path = (
            ORIGINAL_OUTPUT
            /
            f"{image_path.stem}_original.jpg"
        )

        if not cv2.imwrite(
            str(original_path),
            original,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                95
            ]
        ):

            print(
                "WARNING: Failed to save original image."
            )

        # ====================================================
        # ADAPTIVE ENHANCEMENT
        # ====================================================

        enhanced, method, metrics = (
            enhance_image(
                original
            )
        )

        print(
            "\n[ENHANCEMENT]"
        )

        print(
            f"Brightness : "
            f"{metrics['brightness']:.1f}"
        )

        print(
            f"Median     : "
            f"{metrics['median']:.1f}"
        )

        print(
            f"Contrast   : "
            f"{metrics['contrast']:.1f}"
        )

        print(
            f"Dark ratio : "
            f"{metrics['dark_ratio'] * 100:.1f}%"
        )

        print(
            f"Method     : "
            f"{method}"
        )

        # ====================================================
        # SAVE ENHANCED
        # ====================================================

        enhanced_path = (
            ENHANCED_OUTPUT
            /
            f"{image_path.stem}_enhanced.jpg"
        )

        if not cv2.imwrite(
            str(enhanced_path),
            enhanced,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                95
            ]
        ):

            print(
                "WARNING: Failed to save enhanced image."
            )

        # ====================================================
        # ORIGINAL YOLO
        # ====================================================

        original_result = run_detection(
            model,
            original
        )

        original_detections = (
            get_detections(
                original_result,
                model
            )
        )

        original_score = (
            calculate_detection_score(
                original_detections
            )
        )

        print_detections(
            "ORIGINAL YOLO",
            original_detections
        )

        print(
            f"Original score: "
            f"{original_score:.3f}"
        )

        # ====================================================
        # ENHANCED YOLO
        # ====================================================

        enhanced_result = run_detection(
            model,
            enhanced
        )

        enhanced_detections = (
            get_detections(
                enhanced_result,
                model
            )
        )

        enhanced_score = (
            calculate_detection_score(
                enhanced_detections
            )
        )

        print_detections(
            "ENHANCED YOLO",
            enhanced_detections
        )

        print(
            f"Enhanced score: "
            f"{enhanced_score:.3f}"
        )

        # ====================================================
        # FINAL DECISION
        # ====================================================

        print(
            "\n[FINAL DECISION]"
        )

        # ----------------------------------------------------
        # CASE 1: Neither detects
        # ----------------------------------------------------

        if (
            not original_detections
            and not enhanced_detections
        ):

            final_source = "UNKNOWN"

            final_base = enhanced

            final_detections = []

            statistics["unknown"] += 1

            print(
                "Neither version produced "
                "a confident detection."
            )

            print(
                "FINAL -> UNKNOWN"
            )

        # ----------------------------------------------------
        # CASE 2: Only enhanced detects
        # ----------------------------------------------------

        elif (
            not original_detections
            and enhanced_detections
        ):

            final_source = "ENHANCED"

            final_base = enhanced

            final_detections = (
                enhanced_detections
            )

            statistics["enhanced"] += 1

            print(
                "Original: NO CONFIDENT OBJECT"
            )

            print(
                "Enhanced: OBJECT DETECTED"
            )

            print(
                "FINAL -> ENHANCED"
            )

        # ----------------------------------------------------
        # CASE 3: Only original detects
        # ----------------------------------------------------

        elif (
            original_detections
            and not enhanced_detections
        ):

            final_source = "ORIGINAL"

            final_base = original

            final_detections = (
                original_detections
            )

            statistics["original"] += 1

            print(
                "Original: OBJECT DETECTED"
            )

            print(
                "Enhanced: NO CONFIDENT OBJECT"
            )

            print(
                "FINAL -> ORIGINAL"
            )

        # ----------------------------------------------------
        # CASE 4: Both detect
        # ----------------------------------------------------

        else:

            improvement = (
                enhanced_score
                -
                original_score
            )

            print(
                f"Original score : "
                f"{original_score:.3f}"
            )

            print(
                f"Enhanced score : "
                f"{enhanced_score:.3f}"
            )

            print(
                f"Improvement    : "
                f"{improvement:+.3f}"
            )

            if improvement >= MIN_IMPROVEMENT:

                final_source = "ENHANCED"

                final_base = enhanced

                final_detections = (
                    enhanced_detections
                )

                statistics["enhanced"] += 1

                print(
                    "Enhanced provides "
                    "meaningful improvement."
                )

                print(
                    "FINAL -> ENHANCED"
                )

            else:

                final_source = "ORIGINAL"

                final_base = original

                final_detections = (
                    original_detections
                )

                statistics["original"] += 1

                print(
                    "Enhancement does not provide "
                    "enough improvement."
                )

                print(
                    "FINAL -> ORIGINAL"
                )

        # ====================================================
        # FINAL IMAGE
        # ====================================================

        if not final_detections:

            final_image = (
                create_unknown_image(
                    final_base
                )
            )

        else:

            final_image = (
                draw_final_result(
                    final_base,
                    final_detections,
                    final_source
                )
            )

        # ====================================================
        # SAVE FINAL
        # ====================================================

        final_path = (
            FINAL_OUTPUT
            /
            f"{image_path.stem}_FINAL.jpg"
        )

        if not cv2.imwrite(
            str(final_path),
            final_image,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                95
            ]
        ):

            print(
                "WARNING: Failed to save final image."
            )

        else:

            print(
                "\nFINAL SAVED:"
            )

            print(
                final_path
            )

        processed += 1

    # ========================================================
    # SUMMARY
    # ========================================================

    total = (
        statistics["original"]
        +
        statistics["enhanced"]
        +
        statistics["unknown"]
    )

    print(
        "\n" + "=" * 80
    )

    print(
        "PROCESSING COMPLETE"
    )

    print(
        "=" * 80
    )

    print(
        f"\nRequested images : "
        f"{len(image_files)}"
    )

    print(
        f"Processed        : "
        f"{processed}"
    )

    print(
        f"Failed           : "
        f"{failed}"
    )

    print(
        f"Final classified : "
        f"{total}"
    )

    print(
        f"Original         : "
        f"{statistics['original']}"
    )

    print(
        f"Enhanced         : "
        f"{statistics['enhanced']}"
    )

    print(
        f"Unknown          : "
        f"{statistics['unknown']}"
    )

    print(
        "\nRun folder:"
    )

    print(
        OUTPUT_DIR.resolve()
    )

    print(
        "\nOriginal images:"
    )

    print(
        ORIGINAL_OUTPUT.resolve()
    )

    print(
        "\nEnhanced images:"
    )

    print(
        ENHANCED_OUTPUT.resolve()
    )

    print(
        "\nFinal images:"
    )

    print(
        FINAL_OUTPUT.resolve()
    )

    print(
        "\nDone."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
