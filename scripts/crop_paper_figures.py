from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


ROOT = Path("/Users/pfuchs/mimuw/tgs/pres2")
MAIN_PAPER_PDF = ROOT / "materials" / "womens-connectivity-extreme-networks.pdf"
SUPPLEMENTARY_PDF = ROOT / "materials" / "1501742_sm.pdf"
WORK_DIR = ROOT / "build" / "pdfimages"
SUPPLEMENTARY_WORK_DIR = ROOT / "build" / "sm_pages"
PROCESSED_DIR = ROOT / "assets" / "figures" / "processed"
DISPLAY_PADDING_PX = 150


@dataclass(frozen=True)
class CropTarget:
    output_name: str
    source_name: str
    description: str
    source_subdir: str = "pdfimages"

    def source_path(self, root: Path) -> Path:
        return root / "build" / self.source_subdir / self.source_name

    def output_path(self, root: Path) -> Path:
        return root / "assets" / "figures" / "processed" / self.output_name


MAIN_TARGET_SPECS: tuple[CropTarget, ...] = (
    CropTarget("1A-1.png", "img-000.png", "Figure 1A, top panel: daily online network snapshot."),
    CropTarget("1A-2.png", "img-000.png", "Figure 1A, middle panel: daily online network snapshot."),
    CropTarget("1A-3.png", "img-000.png", "Figure 1A, bottom panel: daily online network snapshot."),
    CropTarget("1B.png", "img-000.png", "Figure 1B: online betweenness time series."),
    CropTarget("1C.png", "img-000.png", "Figure 1C: degree centrality with gender-shuffle null."),
    CropTarget("1D-left.png", "img-000.png", "Figure 1D left: high-degree centrality star cartoon."),
    CropTarget("1D-right.png", "img-000.png", "Figure 1D right: low-degree brokerage cartoon."),
    CropTarget("2A-1.png", "img-001.png", "Figure 2A, top panel: PIRA network snapshot (pre-reorganisation)."),
    CropTarget("2A-2.png", "img-001.png", "Figure 2A, middle panel: PIRA network snapshot (reorganisation)."),
    CropTarget("2A-3.png", "img-001.png", "Figure 2A, bottom panel: PIRA network snapshot (post-reorganisation)."),
    CropTarget("2B.png", "img-001.png", "Figure 2B: IED attacks over time."),
    CropTarget("2C.png", "img-001.png", "Figure 2C: PIRA betweenness trend."),
    CropTarget("2D.png", "img-001.png", "Figure 2D: PIRA degree trend."),
    CropTarget("2E.png", "img-001.png", "Figure 2E: model comparison panels on the right."),
    CropTarget("3A.png", "img-002.png", "Figure 3A: ISIS group lifetime vs women/men ratio."),
    CropTarget("3B.png", "img-002.png", "Figure 3B: PIRA lifetime connections comparison."),
)

SUPPLEMENTARY_TARGET_SPECS: tuple[CropTarget, ...] = (
    CropTarget("S1.png", "page-2.png", "Figure S1: BC worked example with a woman-centered shortest-path matrix.", source_subdir="sm_pages"),
    CropTarget("S2.png", "page-4.png", "Figure S2: PIRA productivity rises while actor counts fall.", source_subdir="sm_pages"),
    CropTarget("S3.png", "page-5.png", "Figure S3: generative multi-agent model schematic.", source_subdir="sm_pages"),
    CropTarget("S4.png", "page-7.png", "Figure S4: CCDF of women's betweenness centrality.", source_subdir="sm_pages"),
    CropTarget("S5.png", "page-9.png", "Figure S5: BC broken down by operational role.", source_subdir="sm_pages"),
)


def build_default_targets(root: Path, collection: str = "main") -> list[CropTarget]:
    del root
    if collection == "main":
        return list(MAIN_TARGET_SPECS)
    if collection == "supplementary":
        return list(SUPPLEMENTARY_TARGET_SPECS)
    raise ValueError(f"Unsupported collection: {collection}")


def pending_targets(
    targets: list[CropTarget],
    processed_dir: Path,
    skip_existing: bool = True,
) -> list[CropTarget]:
    if not skip_existing:
        return targets

    return [
        target
        for target in targets
        if not (processed_dir / target.output_name).exists()
    ]


def ensure_source_images(root: Path, collection: str) -> None:
    if collection == "main":
        work_dir = root / "build" / "pdfimages"
        expected = [work_dir / "img-000.png", work_dir / "img-001.png", work_dir / "img-002.png"]
        if all(path.exists() for path in expected):
            return

        work_dir.mkdir(parents=True, exist_ok=True)
        for old in work_dir.glob("img-*.png"):
            old.unlink()

        subprocess.run(
            ["pdfimages", "-png", str(root / "materials" / "womens-connectivity-extreme-networks.pdf"), str(work_dir / "img")],
            check=True,
        )
        return

    if collection == "supplementary":
        work_dir = root / "build" / "sm_pages"
        expected = [work_dir / f"page-{n}.png" for n in (2, 4, 5, 7, 9)]
        if all(path.exists() for path in expected):
            return

        work_dir.mkdir(parents=True, exist_ok=True)
        for old in work_dir.glob("page-*.png"):
            old.unlink()

        subprocess.run(
            ["pdftoppm", "-png", str(root / "materials" / "1501742_sm.pdf"), str(work_dir / "page")],
            check=True,
        )
        return

    raise ValueError(f"Unsupported collection: {collection}")


def fit_for_display(
    image: np.ndarray,
    max_width: int = 1400,
    max_height: int = 950,
) -> tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    scale = min(max_width / width, max_height / height, 1.0)
    if scale == 1.0:
        return image.copy(), 1.0

    resized = cv2.resize(
        image,
        (int(width * scale), int(height * scale)),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def crop_from_rect(image: np.ndarray, rect: tuple[int, int, int, int], scale: float) -> np.ndarray:
    x, y, width, height = rect
    if width <= 0 or height <= 0:
        return image[0:0, 0:0]

    left = max(0, int(round(x / scale)))
    top = max(0, int(round(y / scale)))
    right = min(image.shape[1], int(round((x + width) / scale)))
    bottom = min(image.shape[0], int(round((y + height) / scale)))
    return image[top:bottom, left:right]


def add_white_padding(image: np.ndarray, padding_px: int = DISPLAY_PADDING_PX) -> np.ndarray:
    if image.size == 0 or padding_px <= 0:
        return image

    return cv2.copyMakeBorder(
        image,
        padding_px,
        padding_px,
        padding_px,
        padding_px,
        borderType=cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )


def show_preview(name: str, crop: np.ndarray) -> None:
    preview, _ = fit_for_display(crop, max_width=900, max_height=700)
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, preview)


def show_message(name: str, lines: list[str]) -> None:
    canvas = np.full((220, 900, 3), 255, dtype=np.uint8)
    y = 50
    for line in lines:
        cv2.putText(
            canvas,
            line,
            (30, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )
        y += 40
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, canvas)


def wait_for_action(valid_keys: set[str]) -> str:
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == 255:
            continue

        char = chr(key).lower()
        if char in valid_keys:
            return char


def run_crop_session(root: Path, collection: str, skip_existing: bool, start_from: str | None) -> int:
    ensure_source_images(root, collection)
    processed_dir = root / "assets" / "figures" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    targets = build_default_targets(root, collection=collection)
    queue = pending_targets(targets, processed_dir, skip_existing=skip_existing)
    if start_from is not None:
        queue = [target for target in queue if target.output_name >= start_from]

    if not queue:
        print("No pending targets. Use --include-existing to re-crop saved outputs.")
        return 0

    index = 0
    while 0 <= index < len(queue):
        target = queue[index]
        source = cv2.imread(str(target.source_path(root)))
        if source is None:
            raise FileNotFoundError(f"Could not load source image {target.source_path(root)}")

        padded_source = add_white_padding(source)
        display_source, scale = fit_for_display(padded_source)
        window_name = f"Crop {index + 1}/{len(queue)} - {target.output_name}"
        print()
        print(f"[{index + 1}/{len(queue)}] {target.output_name}")
        print(target.description)
        print("Drag a rectangle and press Enter/Space to confirm it. Press c in the ROI tool to cancel.")

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, display_source)
        rect = cv2.selectROI(window_name, display_source, showCrosshair=True, fromCenter=False)
        crop = crop_from_rect(padded_source, rect, scale)

        if crop.size == 0:
            show_message(
                "Crop Action",
                [
                    "No crop selected.",
                    "r = retry   n = skip target   p = previous   q = quit",
                ],
            )
            action = wait_for_action({"r", "n", "p", "q"})
            cv2.destroyWindow("Crop Action")
            cv2.destroyWindow(window_name)
            if action == "r":
                continue
            if action == "n":
                index += 1
                continue
            if action == "p":
                index = max(0, index - 1)
                continue
            break

        show_preview("Crop Preview", crop)
        show_message(
            "Crop Action",
            [
                target.description,
                f"s = save crop   r = redo   n = skip   p = previous   q = quit   shown source has {DISPLAY_PADDING_PX}px white border",
            ],
        )
        action = wait_for_action({"s", "r", "n", "p", "q"})
        cv2.destroyWindow("Crop Preview")
        cv2.destroyWindow("Crop Action")
        cv2.destroyWindow(window_name)

        if action == "s":
            output_path = target.output_path(root)
            if not cv2.imwrite(str(output_path), crop):
                raise RuntimeError(f"Failed to write crop to {output_path}")
            print(f"Saved {output_path}")
            index += 1
            continue
        if action == "r":
            continue
        if action == "n":
            index += 1
            continue
        if action == "p":
            index = max(0, index - 1)
            continue
        break

    cv2.destroyAllWindows()
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactively crop paper figure panels with an OpenCV UI.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="Project root containing materials/, build/, and assets/ directories.",
    )
    parser.add_argument(
        "--include-existing",
        action="store_true",
        help="Include outputs that already exist in assets/figures/processed.",
    )
    parser.add_argument(
        "--from-target",
        dest="from_target",
        help="Start from this output file name, for example 2C.png.",
    )
    parser.add_argument(
        "--collection",
        choices=("main", "supplementary"),
        default="main",
        help="Which figure collection to crop.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return run_crop_session(
        root=args.root,
        collection=args.collection,
        skip_existing=not args.include_existing,
        start_from=args.from_target,
    )


if __name__ == "__main__":
    sys.exit(main())
