#!/usr/bin/env python3
"""Render OpenPose JSON files to *_depth.png with Blender.

This script is meant to be executed by Blender, not by normal Python:

    blender -b --factory-startup --python scripts/blender_render_pose_depths.py -- --root "C:\\...\\models\\openpose"

OpenPose JSON files only contain 2D keypoints plus confidence values. They do
not contain true Z coordinates. For that input this script builds a small 3D
capsule proxy from the keypoints and renders a grayscale depth-like map. If you
need physically correct human depth, render a posed body mesh instead.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_OPENPOSE_ROOT = Path(
    r"C:\EasyDiffusion\stable-diffusion\stable-diffusion-webui\models\openpose"
)

POSE_FILE_SUFFIX_RE = re.compile(
    r"_(dup\d+|duplicate\d*|copy\d*|bone_structure_full|bone_structure|openposefull|openposehand|openpose|"
    r"depthhand|normalhand|cannyhand|line_art|lineart|linart|canny|depth|normal)"
    r"(_[a-z0-9]+)?$",
    re.IGNORECASE,
)

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

COCO_17_LIMBS = [
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (0, 1),
    (1, 3),
    (0, 2),
    (2, 4),
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

BASE_DEPTH_BY_INDEX = {
    0: -18.0,
    1: 0.0,
    2: 4.0,
    3: -8.0,
    4: -18.0,
    5: 4.0,
    6: -8.0,
    7: -18.0,
    8: 10.0,
    9: 2.0,
    10: -8.0,
    11: 10.0,
    12: 2.0,
    13: -8.0,
    14: -20.0,
    15: -20.0,
    16: -16.0,
    17: -16.0,
}

Point = Tuple[float, float, float]


def blender_args() -> List[str]:
    if "--" not in sys.argv:
        return []
    return sys.argv[sys.argv.index("--") + 1 :]


def strip_pose_file_suffix(stem: str) -> str:
    base = stem.strip()
    while True:
        match = POSE_FILE_SUFFIX_RE.search(base)
        if not match:
            return base.strip()
        base = base[: match.start()].rstrip()


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def point_triplet(value: Any) -> Optional[Point]:
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
        values = [float(item) for item in value]
        if len(values) >= 34:
            if len(values) % 3 == 0:
                return values
            if len(values) % 2 == 0:
                out: List[float] = []
                for index in range(0, len(values), 2):
                    out.extend([values[index], values[index + 1], 1.0])
                return out

    points: List[float] = []
    for item in value:
        triplet = point_triplet(item)
        if triplet is None:
            return None
        points.extend(triplet)
    return points if points else None


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
            if not isinstance(person, dict):
                continue
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


def normalize_to_openpose18(flat: Sequence[float]) -> List[float]:
    point_count = len(flat) // 3
    if point_count == 18:
        return [float(value) for value in flat[: 18 * 3]]
    if point_count >= 25:
        source = [flat[index * 3 : index * 3 + 3] for index in range(point_count)]
        out = [[0.0, 0.0, 0.0] for _ in range(18)]
        for source_idx, target_idx in BODY25_TO_OPENPOSE18.items():
            if source_idx < point_count:
                out[target_idx] = [float(value) for value in source[source_idx]]
        return [value for point in out for value in point]
    if point_count == 17:
        source = [flat[index * 3 : index * 3 + 3] for index in range(point_count)]
        out = [[0.0, 0.0, 0.0] for _ in range(18)]
        for source_idx, target_idx in COCO17_TO_OPENPOSE18.items():
            if source_idx < point_count:
                out[target_idx] = [float(value) for value in source[source_idx]]
        return [value for point in out for value in point]
    return [float(value) for value in flat]


def reshape_points(flat: Sequence[float]) -> List[Point]:
    points = []
    for index in range(0, len(flat) - 2, 3):
        points.append((float(flat[index]), float(flat[index + 1]), float(flat[index + 2])))
    return points


def infer_dimensions(payload: Any, default_width: int, default_height: int, companion: Optional[Path]) -> Tuple[int, int]:
    data = payload[0] if isinstance(payload, list) and payload and isinstance(payload[0], dict) else payload
    width = height = None
    if isinstance(data, dict):
        width = data.get("canvas_width", data.get("width"))
        height = data.get("canvas_height", data.get("height"))

    if (not is_number(width) or not is_number(height)) and companion:
        image_size = read_image_size_with_blender(companion)
        if image_size:
            width, height = image_size

    width = int(width) if is_number(width) and int(width) > 0 else default_width
    height = int(height) if is_number(height) and int(height) > 0 else default_height
    return width, height


def read_image_size_with_blender(path: Path) -> Optional[Tuple[int, int]]:
    if not path.exists():
        return None
    import bpy  # type: ignore

    image = None
    try:
        image = bpy.data.images.load(str(path), check_existing=False)
        return int(image.size[0]), int(image.size[1])
    except Exception:
        return None
    finally:
        if image is not None:
            bpy.data.images.remove(image)


def companion_image_for(json_path: Path, base_name: str) -> Optional[Path]:
    for suffix in ("_bone_structure.png", "_bone_structure_full.png", ".png"):
        candidate = json_path.parent / f"{base_name}{suffix}"
        if candidate.exists():
            return candidate
    return None


def has_matching_depth(folder: Path, base_name: str) -> bool:
    exact = folder / f"{base_name}_depth.png"
    if exact.exists():
        return True
    prefix = f"{base_name}_depth_".lower()
    for candidate in folder.glob("*.png"):
        if candidate.stem.lower().startswith(prefix):
            return True
    return False


def iter_pose_jsons(root: Path, include_plain_json: bool) -> Iterable[Path]:
    pattern = "*.json" if include_plain_json else "*_openpose.json"
    for path in sorted(root.rglob(pattern), key=lambda p: str(p).lower()):
        if path.is_file():
            yield path


def visible_points(points: Sequence[Point]) -> List[Tuple[int, Point]]:
    return [(index, point) for index, point in enumerate(points) if point[2] > 0 and point[0] > 0 and point[1] > 0]


def torso_center(points: Sequence[Point]) -> Tuple[float, float]:
    preferred = [1, 2, 5, 8, 11]
    visible = [points[index] for index in preferred if index < len(points) and points[index][2] > 0]
    if not visible:
        visible = [point for _, point in visible_points(points)]
    if not visible:
        return 0.0, 0.0
    return sum(point[0] for point in visible) / len(visible), sum(point[1] for point in visible) / len(visible)


def synthetic_depth_values(points: Sequence[Point], width: int, height: int) -> Dict[int, float]:
    center_x, center_y = torso_center(points)
    diagonal = max(1.0, math.hypot(width, height))
    values: Dict[int, float] = {}
    for index, point in visible_points(points):
        distance = math.hypot(point[0] - center_x, point[1] - center_y) / diagonal
        base = BASE_DEPTH_BY_INDEX.get(index, 0.0)
        values[index] = base - min(22.0, distance * 46.0)
    return values


def gray_for_depth(depth_value: float, min_depth: float, max_depth: float, near_gray: int, far_gray: int) -> int:
    if max_depth <= min_depth:
        return near_gray
    normalized = (depth_value - min_depth) / (max_depth - min_depth)
    return int(round(near_gray + (far_gray - near_gray) * normalized))


def setup_scene(width: int, height: int, camera_distance: float) -> None:
    import bpy  # type: ignore
    from mathutils import Vector  # type: ignore

    scene = bpy.context.scene
    engine_items = scene.render.bl_rna.properties["engine"].enum_items
    supported_engines = {item.identifier for item in engine_items}
    if "BLENDER_EEVEE_NEXT" in supported_engines:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
    elif "BLENDER_EEVEE" in supported_engines:
        scene.render.engine = "BLENDER_EEVEE"
    else:
        scene.render.engine = "BLENDER_WORKBENCH"
    scene.render.resolution_x = int(width)
    scene.render.resolution_y = int(height)
    scene.render.resolution_percentage = 100
    scene.frame_set(1)
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0
    scene.view_settings.gamma = 1
    scene.display_settings.display_device = "sRGB"
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    scene.render.image_settings.compression = 15
    if hasattr(scene, "eevee") and hasattr(scene.eevee, "taa_render_samples"):
        scene.eevee.taa_render_samples = 32

    world = scene.world or bpy.data.worlds.new("World")
    scene.world = world
    world.color = (0.0, 0.0, 0.0)

    camera = bpy.data.objects.get("DepthCamera")
    if camera is None:
        camera_data = bpy.data.cameras.new("DepthCamera")
        camera = bpy.data.objects.new("DepthCamera", camera_data)
        bpy.context.collection.objects.link(camera)
    camera.location = (width / 2.0, -camera_distance, height / 2.0)
    target = Vector((width / 2.0, 0.0, height / 2.0))
    direction = target - Vector(camera.location)
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = float(height)
    camera.data.clip_start = 0.1
    camera.data.clip_end = camera_distance * 3.0
    scene.camera = camera


def clear_pose_objects() -> None:
    import bpy  # type: ignore

    for obj in list(bpy.context.scene.objects):
        if obj.name.startswith("PoseDepth_"):
            bpy.data.objects.remove(obj, do_unlink=True)


def material_for_gray(gray: int):
    import bpy  # type: ignore

    gray = max(0, min(255, int(gray)))
    name = f"PoseDepth_gray_{gray:03d}"
    existing = bpy.data.materials.get(name)
    if existing:
        return existing

    value = gray / 255.0
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    for node in list(nodes):
        nodes.remove(node)
    emission = nodes.new(type="ShaderNodeEmission")
    emission.inputs["Color"].default_value = (value, value, value, 1.0)
    emission.inputs["Strength"].default_value = 1.0
    output = nodes.new(type="ShaderNodeOutputMaterial")
    material.node_tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return material


def world_point(point: Point, depth_value: float, height: int) -> Tuple[float, float, float]:
    return point[0], depth_value, height - point[1]


def add_cylinder_between(start: Tuple[float, float, float], end: Tuple[float, float, float], radius: float, gray: int) -> None:
    import bpy  # type: ignore
    from mathutils import Vector  # type: ignore

    start_v = Vector(start)
    end_v = Vector(end)
    direction = end_v - start_v
    length = direction.length
    if length <= 0.001:
        return

    mid = start_v + direction * 0.5
    bpy.ops.mesh.primitive_cylinder_add(vertices=20, radius=radius, depth=length, location=mid)
    obj = bpy.context.object
    obj.name = "PoseDepth_limb"
    obj.rotation_euler = direction.to_track_quat("Z", "Y").to_euler()
    obj.data.materials.append(material_for_gray(gray))


def add_sphere(location: Tuple[float, float, float], radius: float, gray: int) -> None:
    import bpy  # type: ignore

    bpy.ops.mesh.primitive_uv_sphere_add(segments=20, ring_count=10, radius=radius, location=location)
    obj = bpy.context.object
    obj.name = "PoseDepth_joint"
    obj.data.materials.append(material_for_gray(gray))


def create_pose_proxy(
    people: Sequence[List[float]],
    width: int,
    height: int,
    radius_px: float,
    near_gray: int,
    far_gray: int,
) -> int:
    object_count = 0

    for person in people:
        points = reshape_points(normalize_to_openpose18(person))
        if not visible_points(points):
            continue

        depth_values = synthetic_depth_values(points, width, height)
        if not depth_values:
            continue
        min_depth = min(depth_values.values())
        max_depth = max(depth_values.values())

        for start_index, end_index in OPENPOSE_18_LIMBS:
            if start_index >= len(points) or end_index >= len(points):
                continue
            start = points[start_index]
            end = points[end_index]
            if start[2] <= 0 or end[2] <= 0:
                continue
            start_depth = depth_values.get(start_index, 0.0)
            end_depth = depth_values.get(end_index, 0.0)
            average_depth = (start_depth + end_depth) * 0.5
            gray = gray_for_depth(average_depth, min_depth, max_depth, near_gray, far_gray)
            add_cylinder_between(
                world_point(start, start_depth, height),
                world_point(end, end_depth, height),
                radius_px,
                gray,
            )
            object_count += 1

        for index, point in visible_points(points):
            depth_value = depth_values.get(index, 0.0)
            gray = gray_for_depth(depth_value, min_depth, max_depth, near_gray, far_gray)
            add_sphere(world_point(point, depth_value, height), radius_px * 1.25, gray)
            object_count += 1

    return object_count


def render_depth(json_path: Path, output_path: Path, args: argparse.Namespace) -> bool:
    import bpy  # type: ignore

    with json_path.open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)

    base_name = strip_pose_file_suffix(json_path.stem)
    companion = companion_image_for(json_path, base_name)
    width, height = infer_dimensions(payload, args.default_width, args.default_height, companion)
    people = extract_people_keypoints(payload)
    if not people:
        print(f"SKIP no keypoints: {json_path}")
        return False

    if args.radius_px > 0:
        radius_px = args.radius_px
    else:
        radius_px = max(2.0, min(width, height) * args.radius_scale)

    clear_pose_objects()
    setup_scene(width, height, args.camera_distance)
    object_count = create_pose_proxy(people, width, height, radius_px, args.near_gray, args.far_gray)
    if object_count == 0:
        print(f"SKIP no visible proxy geometry: {json_path}")
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.context.scene.render.filepath = str(output_path)
    bpy.ops.render.render(write_still=True)
    print(f"WROTE {output_path}")
    return True


def parse_args() -> argparse.Namespace:
    env_root = os.environ.get("OPENPOSE_MODELS_PATH")
    default_root = Path(env_root) if env_root else DEFAULT_OPENPOSE_ROOT

    parser = argparse.ArgumentParser(description="Render missing *_depth.png files from OpenPose JSONs using Blender.")
    parser.add_argument("--root", type=Path, default=default_root, help="OpenPose dataset root.")
    parser.add_argument("--include-plain-json", action="store_true", help="Scan all JSON files instead of *_openpose.json.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing depth files.")
    parser.add_argument("--limit", type=int, default=0, help="Maximum number of files to render. 0 means no limit.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned outputs without rendering.")
    parser.add_argument("--default-width", type=int, default=768, help="Canvas width fallback when JSON/image dimensions are missing.")
    parser.add_argument("--default-height", type=int, default=768, help="Canvas height fallback when JSON/image dimensions are missing.")
    parser.add_argument("--radius-px", type=float, default=0.0, help="Fixed skeleton capsule radius in pixels.")
    parser.add_argument("--radius-scale", type=float, default=0.008, help="Capsule radius as fraction of min(width, height).")
    parser.add_argument("--near-gray", type=int, default=238, help="Grayscale value for nearest proxy parts.")
    parser.add_argument("--far-gray", type=int, default=118, help="Grayscale value for farthest proxy parts.")
    parser.add_argument("--camera-distance", type=float, default=1000.0, help="Camera distance in Blender units.")
    return parser.parse_args(blender_args())


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    if not root.exists():
        print(f"OpenPose root does not exist: {root}", file=sys.stderr)
        return 2

    candidates: List[Tuple[Path, Path]] = []
    for json_path in iter_pose_jsons(root, args.include_plain_json):
        base_name = strip_pose_file_suffix(json_path.stem)
        if not base_name:
            continue
        output_path = json_path.parent / f"{base_name}_depth.png"
        if not args.overwrite and has_matching_depth(json_path.parent, base_name):
            continue
        candidates.append((json_path, output_path))

    if args.limit > 0:
        candidates = candidates[: args.limit]

    print(f"Depth render candidates: {len(candidates)}")
    if args.dry_run:
        for json_path, output_path in candidates:
            print(f"DRY {json_path} -> {output_path}")
        return 0

    written = 0
    failed = 0
    for index, (json_path, output_path) in enumerate(candidates, start=1):
        print(f"[{index}/{len(candidates)}] {json_path}")
        try:
            if render_depth(json_path, output_path, args):
                written += 1
            else:
                failed += 1
        except Exception as exc:
            failed += 1
            print(f"FAILED {json_path}: {exc}", file=sys.stderr)

    clear_pose_objects()
    print(f"Done. Written: {written}. Failed/skipped: {failed}.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
