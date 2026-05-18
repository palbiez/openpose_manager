#!/usr/bin/env python3
"""Import downloaded pose collections into the OpenPose Manager layout.

The script accepts common downloaded pose formats and writes OPM-compatible
files:

    <output_root>/<pose>/<gender>/<variant>/<subpose>/<base>_openpose.json
    <output_root>/<pose>/<gender>/<variant>/<subpose>/<base>_bone_structure.png

Supported inputs include:
- OpenPose JSON: people[].pose_keypoints_2d
- OpenPose Collection v2 style: [ { people: [...] } ]
- OpenPose Editor/Dynamic Poses style: { width, height, keypoints: [[x, y], ...] }
- Optional preset collection JSONs when --include-presets is used
"""

from __future__ import annotations

import argparse
import binascii
import json
import os
import re
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import zlib


DEFAULT_SOURCE_ENV = "OPENPOSE_IMPORT_SOURCE"
DEFAULT_SOURCE = Path(os.environ[DEFAULT_SOURCE_ENV]) if os.environ.get(DEFAULT_SOURCE_ENV) else None

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
JSON_EXT = ".json"

OPENPOSE_18_LIMBS = [
    (1, 2),
    (2, 3),
    (3, 4),
    (1, 5),
    (5, 6),
    (6, 7),
    (1, 8),
    (8, 9),
    (9, 10),
    (1, 11),
    (11, 12),
    (12, 13),
    (1, 0),
    (0, 14),
    (14, 16),
    (0, 15),
    (15, 17),
]

OPENPOSE_COLORS = [
    (255, 0, 0),
    (255, 85, 0),
    (255, 170, 0),
    (255, 255, 0),
    (170, 255, 0),
    (85, 255, 0),
    (0, 255, 0),
    (0, 255, 85),
    (0, 255, 170),
    (0, 255, 255),
    (0, 170, 255),
    (0, 85, 255),
    (0, 0, 255),
    (85, 0, 255),
    (170, 0, 255),
    (255, 0, 255),
    (255, 0, 170),
    (255, 0, 85),
]

BODY25_TO_OPENPOSE18 = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
    7: 7,
    9: 8,
    10: 9,
    11: 10,
    12: 11,
    13: 12,
    14: 13,
    15: 14,
    16: 15,
    17: 16,
    18: 17,
}

COCO17_TO_OPENPOSE18 = {
    0: 0,
    6: 2,
    8: 3,
    10: 4,
    5: 5,
    7: 6,
    9: 7,
    12: 8,
    14: 9,
    16: 10,
    11: 11,
    13: 12,
    15: 13,
    2: 14,
    1: 15,
    4: 16,
    3: 17,
}


@dataclass
class PoseRecord:
    source_json: Path
    name: str
    payload: Any
    collection: str


@dataclass
class ImportResult:
    imported: int = 0
    skipped: int = 0
    failed: int = 0
    images_written: int = 0


def slugify(value: Any, fallback: str = "pose") -> str:
    text = str(value or "").strip().lower()
    text = text.replace("-", "_").replace(" ", "_")
    text = re.sub(r"[^a-z0-9_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or fallback


def normalize_gender(value: Any, fallback: str = "F") -> str:
    text = str(value or "").strip()
    if not text:
        return fallback
    normalized = text.lower()
    if normalized in {"f", "female", "woman", "women"}:
        return "F"
    if normalized in {"m", "male", "man", "men"}:
        return "M"
    return slugify(text, fallback.lower())


def clean_pose_name(stem: str) -> str:
    name = stem
    for suffix in ("_openpose", "_normalized", "_json", "-json"):
        if name.lower().endswith(suffix):
            name = name[: -len(suffix)]
            break
    return slugify(name)


def default_output_root(source_root: Path) -> Path:
    normalized_name = slugify(source_root.name)
    if normalized_name in {"new_poses", "newposes"}:
        return source_root.parent
    return source_root / "_opm_imported"


def collection_slug(source_root: Path, json_path: Path) -> str:
    try:
        rel = json_path.relative_to(source_root)
    except ValueError:
        return slugify(json_path.parent.name, "collection")
    parts = rel.parts
    if len(parts) > 1:
        return slugify(parts[0], "collection")
    return slugify(json_path.parent.name, "collection")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def point_triplet(value: Any) -> Optional[Tuple[float, float, float]]:
    if isinstance(value, dict):
        x = value.get("x")
        y = value.get("y")
        c = value.get("c", value.get("confidence", value.get("score", 1.0)))
    elif isinstance(value, (list, tuple)) and len(value) >= 2:
        x = value[0]
        y = value[1]
        c = value[2] if len(value) >= 3 else 1.0
    else:
        return None

    if not is_number(x) or not is_number(y):
        return None
    if not is_number(c):
        c = 1.0
    return float(x), float(y), float(c)


def flatten_keypoints(value: Any) -> Optional[List[float]]:
    if not isinstance(value, list) or not value:
        return None

    if all(is_number(item) for item in value):
        flat = [float(item) for item in value]
        if len(flat) % 3 == 0:
            return flat
        if len(flat) % 2 == 0:
            out: List[float] = []
            for index in range(0, len(flat), 2):
                out.extend([flat[index], flat[index + 1], 1.0])
            return out
        return None

    out = []
    for item in value:
        triplet = point_triplet(item)
        if triplet is None:
            return None
        out.extend(triplet)
    return out if out else None


def looks_like_pose_payload(data: Any) -> bool:
    if isinstance(data, list):
        return bool(data) and any(looks_like_pose_payload(item) for item in data[:3])
    if not isinstance(data, dict):
        return False
    if isinstance(data.get("people"), list):
        return True
    for key in ("pose_keypoints_2d", "keypoints"):
        if flatten_keypoints(data.get(key)) is not None:
            return True
    return False


def iter_records(json_path: Path, source_root: Path, include_presets: bool) -> Iterable[PoseRecord]:
    data = load_json(json_path)
    collection = collection_slug(source_root, json_path)

    if json_path.name.lower() == "presets.json" and not include_presets:
        return

    if looks_like_pose_payload(data):
        yield PoseRecord(
            source_json=json_path,
            name=clean_pose_name(json_path.stem),
            payload=data,
            collection=collection,
        )
        return

    if isinstance(data, dict):
        for key, value in sorted(data.items(), key=lambda item: str(item[0]).lower()):
            if looks_like_pose_payload(value):
                yield PoseRecord(
                    source_json=json_path,
                    name=clean_pose_name(str(key)),
                    payload=value,
                    collection=collection,
                )


def extract_people_keypoints(payload: Any) -> List[List[float]]:
    if isinstance(payload, list):
        people: List[List[float]] = []
        for item in payload:
            people.extend(extract_people_keypoints(item))
        return people

    if not isinstance(payload, dict):
        return []

    raw_people = payload.get("people")
    if isinstance(raw_people, list):
        people = []
        for person in raw_people:
            if isinstance(person, dict):
                flat = flatten_keypoints(person.get("pose_keypoints_2d"))
                if flat is None:
                    flat = flatten_keypoints(person.get("keypoints"))
                if flat:
                    people.append(flat)
        return people

    for key in ("pose_keypoints_2d", "keypoints"):
        flat = flatten_keypoints(payload.get(key))
        if flat:
            return [flat]

    return []


def infer_dimensions(payload: Any, companion: Optional[Path], default_size: int) -> Tuple[int, int]:
    data = payload[0] if isinstance(payload, list) and payload and isinstance(payload[0], dict) else payload
    width = height = None
    if isinstance(data, dict):
        width = data.get("canvas_width", data.get("width"))
        height = data.get("canvas_height", data.get("height"))

    if (not is_number(width) or not is_number(height)) and companion:
        image_size = read_image_size(companion)
        if image_size is not None:
            width, height = image_size

    width = int(width) if is_number(width) and int(width) > 0 else default_size
    height = int(height) if is_number(height) and int(height) > 0 else default_size
    return width, height


def read_image_size(path: Path) -> Optional[Tuple[int, int]]:
    try:
        from PIL import Image
    except Exception:
        return None

    try:
        with Image.open(path) as image:
            return int(image.width), int(image.height)
    except Exception:
        return None


def reshape_points(flat: Sequence[float]) -> List[List[float]]:
    count = len(flat) // 3
    return [[float(flat[i * 3]), float(flat[i * 3 + 1]), float(flat[i * 3 + 2])] for i in range(count)]


def normalize_layout_to_openpose18(flat: Sequence[float]) -> List[float]:
    points = reshape_points(flat)
    count = len(points)
    if count == 18:
        return [value for point in points for value in point]

    out = [[0.0, 0.0, 0.0] for _ in range(18)]
    if count >= 25:
        for source_idx, target_idx in BODY25_TO_OPENPOSE18.items():
            if source_idx < count:
                out[target_idx] = points[source_idx]
        return [value for point in out for value in point]

    if count == 17:
        for source_idx, target_idx in COCO17_TO_OPENPOSE18.items():
            out[target_idx] = points[source_idx]
        left_shoulder = points[5]
        right_shoulder = points[6]
        if left_shoulder[2] > 0 and right_shoulder[2] > 0:
            out[1] = [
                (left_shoulder[0] + right_shoulder[0]) / 2.0,
                (left_shoulder[1] + right_shoulder[1]) / 2.0,
                min(left_shoulder[2], right_shoulder[2]),
            ]
        return [value for point in out for value in point]

    if count > 18:
        return [value for point in points[:18] for value in point]

    return [value for point in points for value in point]


def denormalize_if_needed(flat: Sequence[float], width: int, height: int) -> List[float]:
    points = reshape_points(flat)
    valid = [point for point in points if point[2] > 0]
    if not valid:
        return [value for point in points for value in point]

    max_x = max(point[0] for point in valid)
    max_y = max(point[1] for point in valid)
    if max_x <= 1.5 and max_y <= 1.5:
        scaled = [[point[0] * width, point[1] * height, point[2]] for point in points]
        return [value for point in scaled for value in point]
    return [value for point in points for value in point]


def classify_pose(name: str, default_pose: str, default_subpose: str) -> Tuple[str, str]:
    token = slugify(name)
    if token.startswith("standing") or token.startswith("tpose"):
        return "standing", "tpose" if token.startswith("tpose") else "standing"
    if token.startswith("sitting"):
        return "sitting", "sitting"
    if token.startswith("laying") or token.startswith("lying"):
        return "lying", "lying"
    if token.startswith("kneeling"):
        return "kneeling", "kneeling"
    if token.startswith("squatting"):
        return "squatting", "squatting"
    if token.startswith("crouching"):
        return "squatting", "crouching"
    if token.startswith("crawling") or token.startswith("all_fours"):
        return "all_fours", "crawling"
    if token.startswith("jumping"):
        return "action", "jumping"
    if token.startswith("running"):
        return "action", "running"
    if token.startswith("walking"):
        return "action", "walking"
    if token.startswith("dance") or token.startswith("dancing"):
        return "action", "dancing"
    if token.startswith("flexing"):
        return "standing", "flexing"
    if "dynamicpose" in token or token.isdigit():
        return "action", "dynamic_pose"
    return slugify(default_pose, "unknown"), slugify(default_subpose, "imported")


def find_companion_image(json_path: Path, record_name: str) -> Optional[Path]:
    folder = json_path.parent
    clean = clean_pose_name(record_name)
    source_stem = clean_pose_name(json_path.stem)
    prefixes = [clean, source_stem]

    if source_stem.endswith("_json"):
        prefixes.append(source_stem[: -len("_json")])
    if source_stem.endswith("-json"):
        prefixes.append(source_stem[: -len("-json")])

    if source_stem.isdigit() or clean.isdigit():
        prefixes.append(str(int(source_stem or clean)))

    prefixes = [slugify(prefix) for prefix in prefixes if prefix]
    seen = set()
    prefixes = [prefix for prefix in prefixes if not (prefix in seen or seen.add(prefix))]

    files = [path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTS]

    def score(path: Path) -> Optional[Tuple[int, str]]:
        stem = slugify(path.stem)
        lower_name = path.name.lower()
        if "preview" in lower_name:
            return None
        for prefix in prefixes:
            if stem == prefix:
                return 0, path.name.lower()
            if stem == f"{prefix}_pose" or stem == f"{prefix}-pose":
                return 1, path.name.lower()
            if stem.startswith(prefix + "_") or stem.startswith(prefix + "-"):
                return 2, path.name.lower()
        return None

    scored = []
    for path in files:
        item_score = score(path)
        if item_score is not None:
            scored.append((item_score, path))
    if not scored:
        return None
    scored.sort(key=lambda item: item[0])
    return scored[0][1]


def write_bone_image(
    target: Path,
    people: Sequence[List[float]],
    width: int,
    height: int,
    companion: Optional[Path],
    render_bone: bool,
    overwrite: bool,
) -> bool:
    if target.exists() and not overwrite:
        return False

    target.parent.mkdir(parents=True, exist_ok=True)
    if companion and not render_bone:
        if companion.suffix.lower() == ".png":
            shutil.copy2(companion, target)
            return True
        if convert_image_to_png(companion, target):
            return True

    return render_bone_image(target, people, width, height)


def convert_image_to_png(source: Path, target: Path) -> bool:
    try:
        from PIL import Image
    except Exception:
        return False

    try:
        with Image.open(source) as image:
            image.convert("RGB").save(target, format="PNG")
        return True
    except Exception:
        return False


def render_bone_image(target: Path, people: Sequence[List[float]], width: int, height: int) -> bool:
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return render_bone_image_basic(target, people, width, height)

    try:
        image = Image.new("RGB", (width, height), (0, 0, 0))
        draw = ImageDraw.Draw(image)
        for person in people:
            points = reshape_points(person)
            for limb_index, (start_index, end_index) in enumerate(OPENPOSE_18_LIMBS):
                if start_index >= len(points) or end_index >= len(points):
                    continue
                start = points[start_index]
                end = points[end_index]
                if start[2] <= 0 or end[2] <= 0:
                    continue
                color = OPENPOSE_COLORS[limb_index % len(OPENPOSE_COLORS)]
                draw.line(
                    [(round(start[0]), round(start[1])), (round(end[0]), round(end[1]))],
                    fill=color,
                    width=max(1, width // 160),
                )
            radius = max(2, width // 180)
            for point_index, point in enumerate(points):
                if point[2] <= 0:
                    continue
                x, y = round(point[0]), round(point[1])
                color = OPENPOSE_COLORS[point_index % len(OPENPOSE_COLORS)]
                draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
        image.save(target, format="PNG")
        return True
    except Exception:
        return render_bone_image_basic(target, people, width, height)


def set_pixel(pixels: bytearray, width: int, height: int, x: int, y: int, color: Tuple[int, int, int]) -> None:
    if x < 0 or y < 0 or x >= width or y >= height:
        return
    offset = (y * width + x) * 3
    pixels[offset : offset + 3] = bytes(color)


def draw_disk(
    pixels: bytearray,
    width: int,
    height: int,
    cx: int,
    cy: int,
    radius: int,
    color: Tuple[int, int, int],
) -> None:
    radius = max(1, radius)
    rr = radius * radius
    for y in range(cy - radius, cy + radius + 1):
        for x in range(cx - radius, cx + radius + 1):
            dx = x - cx
            dy = y - cy
            if dx * dx + dy * dy <= rr:
                set_pixel(pixels, width, height, x, y, color)


def draw_line_basic(
    pixels: bytearray,
    width: int,
    height: int,
    start: Tuple[int, int],
    end: Tuple[int, int],
    color: Tuple[int, int, int],
    thickness: int,
) -> None:
    x0, y0 = start
    x1, y1 = end
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    radius = max(1, thickness // 2)

    while True:
        draw_disk(pixels, width, height, x0, y0, radius, color)
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


def png_chunk(kind: bytes, data: bytes) -> bytes:
    checksum = binascii.crc32(kind)
    checksum = binascii.crc32(data, checksum) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", checksum)


def write_png_rgb(path: Path, width: int, height: int, pixels: bytes) -> None:
    rows = []
    stride = width * 3
    for y in range(height):
        rows.append(b"\x00" + pixels[y * stride : (y + 1) * stride])
    raw = b"".join(rows)
    payload = [
        b"\x89PNG\r\n\x1a\n",
        png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)),
        png_chunk(b"IDAT", zlib.compress(raw, 9)),
        png_chunk(b"IEND", b""),
    ]
    path.write_bytes(b"".join(payload))


def render_bone_image_basic(target: Path, people: Sequence[List[float]], width: int, height: int) -> bool:
    try:
        pixels = bytearray(width * height * 3)
        line_width = max(2, width // 160)
        point_radius = max(3, width // 180)
        for person in people:
            points = reshape_points(person)
            for limb_index, (start_index, end_index) in enumerate(OPENPOSE_18_LIMBS):
                if start_index >= len(points) or end_index >= len(points):
                    continue
                start = points[start_index]
                end = points[end_index]
                if start[2] <= 0 or end[2] <= 0:
                    continue
                color = OPENPOSE_COLORS[limb_index % len(OPENPOSE_COLORS)]
                draw_line_basic(
                    pixels,
                    width,
                    height,
                    (round(start[0]), round(start[1])),
                    (round(end[0]), round(end[1])),
                    color,
                    line_width,
                )
            for point_index, point in enumerate(points):
                if point[2] <= 0:
                    continue
                color = OPENPOSE_COLORS[point_index % len(OPENPOSE_COLORS)]
                draw_disk(pixels, width, height, round(point[0]), round(point[1]), point_radius, color)
        write_png_rgb(target, width, height, bytes(pixels))
        return True
    except Exception:
        return False


def build_output_payload(
    people: Sequence[List[float]],
    width: int,
    height: int,
    record: PoseRecord,
    pose: str,
    subpose: str,
    gender: str,
    variant: str,
) -> Dict[str, Any]:
    return {
        "version": 1.0,
        "canvas_width": width,
        "canvas_height": height,
        "people": [
            {
                "pose_keypoints_2d": list(person),
                "face_keypoints_2d": [],
                "hand_left_keypoints_2d": [],
                "hand_right_keypoints_2d": [],
            }
            for person in people
        ],
        "meta": {
            "source_file": str(record.source_json),
            "source_collection": record.collection,
            "source_name": record.name,
            "pose": pose,
            "gender": gender,
            "variant": variant,
            "subpose": subpose,
            "schema": "opm_import_pose_collections/v1",
        },
    }


def iter_json_files(source_root: Path, output_root: Path) -> Iterable[Path]:
    try:
        output_is_inside_source = output_root.resolve().is_relative_to(source_root.resolve())
    except AttributeError:
        try:
            output_root.resolve().relative_to(source_root.resolve())
            output_is_inside_source = True
        except ValueError:
            output_is_inside_source = False

    for path in sorted(source_root.rglob("*.json"), key=lambda item: str(item).lower()):
        if path.name.lower().endswith("_openpose.json"):
            continue
        if output_is_inside_source and output_root in path.parents:
            continue
        yield path


def import_record(record: PoseRecord, args: argparse.Namespace) -> Tuple[str, Optional[Path], bool]:
    companion = find_companion_image(record.source_json, record.name)
    width, height = infer_dimensions(record.payload, companion, args.default_size)

    raw_people = extract_people_keypoints(record.payload)
    if not raw_people:
        return "failed", None, False

    people = []
    for flat in raw_people:
        converted = normalize_layout_to_openpose18(flat)
        converted = denormalize_if_needed(converted, width, height)
        if len(converted) >= 18 * 3:
            people.append(converted[: 18 * 3])

    if not people:
        return "failed", None, False

    derived_pose, derived_subpose = classify_pose(record.name, args.default_pose, args.default_subpose)
    pose = slugify(args.pose_override or derived_pose)
    subpose = slugify(args.subpose_override or derived_subpose)
    gender = normalize_gender(args.gender, "F")
    variant = slugify(args.variant, "base")

    base_name = slugify(f"{record.collection}_{record.name}")
    out_dir = args.output_root / pose / gender / variant / subpose
    json_path = out_dir / f"{base_name}_openpose.json"
    bone_path = out_dir / f"{base_name}_bone_structure.png"

    if json_path.exists() and not args.overwrite:
        return "skipped", json_path, False

    payload = build_output_payload(people, width, height, record, pose, subpose, gender, variant)
    image_written = False
    if not args.dry_run:
        save_json(json_path, payload)
        if not args.no_images:
            image_written = write_bone_image(
                bone_path,
                people,
                width,
                height,
                companion,
                render_bone=args.render_bone,
                overwrite=args.overwrite,
            )
    elif not args.no_images:
        image_written = True
    return "imported", json_path, image_written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import downloaded pose collections into the OPM dataset layout.")
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help=f"Source folder containing downloaded pose collections. Can also be set via {DEFAULT_SOURCE_ENV}.",
    )
    parser.add_argument("--output-root", type=Path, default=None, help="OpenPose output root. Defaults to the parent of 'new poses'.")
    parser.add_argument("--variant", default="base", help="Dataset variant to write, usually 'base'.")
    parser.add_argument("--gender", default="F", help="Gender token to write into the folder layout.")
    parser.add_argument("--default-pose", default="unknown", help="Pose category for names that cannot be classified.")
    parser.add_argument("--default-subpose", default="imported", help="Subpose for names that cannot be classified.")
    parser.add_argument("--pose-override", default="", help="Force all imported files into one pose category.")
    parser.add_argument("--subpose-override", default="", help="Force all imported files into one subpose category.")
    parser.add_argument("--include-presets", action="store_true", help="Also import presets.json collection entries.")
    parser.add_argument("--render-bone", action="store_true", help="Render bone_structure PNGs from JSON instead of copying companion images.")
    parser.add_argument("--no-images", action="store_true", help="Only write *_openpose.json files.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing imported files.")
    parser.add_argument("--dry-run", action="store_true", help="Report what would be imported without writing files.")
    parser.add_argument("--default-size", type=int, default=768, help="Canvas size fallback when JSON/image dimensions are missing.")
    parser.add_argument("--limit", type=int, default=0, help="Optional maximum number of records to process.")
    parser.add_argument("--verbose", action="store_true", help="Print each imported/skipped file.")
    args = parser.parse_args()

    if args.source is not None:
        args.source = args.source.expanduser().resolve()
    if args.output_root is None:
        if args.source is None:
            args.output_root = None
        else:
            args.output_root = default_output_root(args.source)
    else:
        args.output_root = args.output_root.expanduser().resolve()
    return args


def main() -> int:
    args = parse_args()
    if args.source is None:
        raise SystemExit(f"Source folder is required. Use --source or set {DEFAULT_SOURCE_ENV}.")
    if not args.source.exists():
        raise SystemExit(f"Source folder does not exist: {args.source}")

    result = ImportResult()
    processed = 0

    for json_path in iter_json_files(args.source, args.output_root):
        try:
            records = list(iter_records(json_path, args.source, args.include_presets))
        except Exception as exc:
            result.failed += 1
            if args.verbose:
                print(f"failed: {json_path} ({exc})")
            continue

        for record in records:
            if args.limit and processed >= args.limit:
                break
            status, out_path, image_written = import_record(record, args)
            processed += 1
            if status == "imported":
                result.imported += 1
                if image_written:
                    result.images_written += 1
            elif status == "skipped":
                result.skipped += 1
            else:
                result.failed += 1
            if args.verbose:
                print(f"{status}: {record.source_json} -> {out_path or ''}")
        if args.limit and processed >= args.limit:
            break

    print(
        json.dumps(
            {
                "source": str(args.source),
                "output_root": str(args.output_root),
                "processed": processed,
                "imported": result.imported,
                "skipped": result.skipped,
                "failed": result.failed,
                "images_written": result.images_written,
                "dry_run": args.dry_run,
                "variant": args.variant,
                "gender": args.gender,
            },
            indent=2,
        )
    )
    return 0 if result.failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
