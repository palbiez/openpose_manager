#!/usr/bin/env python3
"""Audit OpenPose depth, bone-structure, and JSON asset groups.

The script checks dataset consistency without renaming files by default:

* groups assets by base name in the OpenPose Manager folder layout
* reports missing depth / bone_structure / openpose JSON files
* validates JSON body keypoint count and canvas dimensions
* reconstructs keypoints from color-coded bone_structure PNGs
* compares reconstructed bone keypoints against existing JSONs
* optionally writes missing *_openpose.json files from bone_structure PNGs

Depth images do not contain keypoints, so depth-vs-bone semantic matching is
reported conservatively through dimensions and JSON/bone cross-match hints.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import re
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


PLUGIN_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OPENPOSE_ROOTS = [
    Path(os.environ["OPENPOSE_MODELS_PATH"]) if os.environ.get("OPENPOSE_MODELS_PATH") else None,
    PLUGIN_ROOT.parent.parent / "models" / "openpose",
    Path.cwd() / "models" / "openpose",
]

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

OPENPOSE_COLORS_RGB = [
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
OPENPOSE_COLOR_CODES = [(red << 16) | (green << 8) | blue for red, green, blue in OPENPOSE_COLORS_RGB]


@dataclass
class ImageInfo:
    width: int
    height: int
    channels: int


@dataclass
class JsonInfo:
    path: Path
    canvas_width: Optional[int]
    canvas_height: Optional[int]
    body_point_count: int
    valid_body_count: int
    keypoints: List[float]
    error: str = ""


@dataclass
class CandidatePoint:
    index: int
    x: float
    y: float
    area: int
    width: int
    height: int
    score: float


@dataclass
class BoneExtraction:
    path: Path
    width: int
    height: int
    keypoints: List[float]
    selected_count: int
    candidate_counts: List[int]
    missing_indices: List[int]
    warnings: List[str] = field(default_factory=list)


@dataclass
class AssetGroup:
    folder: Path
    base_name: str
    files: List[Path] = field(default_factory=list)
    depth_files: List[Path] = field(default_factory=list)
    bone_files: List[Path] = field(default_factory=list)
    bone_full_files: List[Path] = field(default_factory=list)
    json_files: List[Path] = field(default_factory=list)

    @property
    def depth(self) -> Optional[Path]:
        return sorted(self.depth_files, key=lambda p: p.name.lower())[0] if self.depth_files else None

    @property
    def bone(self) -> Optional[Path]:
        if self.bone_files:
            return sorted(self.bone_files, key=lambda p: p.name.lower())[0]
        return sorted(self.bone_full_files, key=lambda p: p.name.lower())[0] if self.bone_full_files else None

    @property
    def json_path(self) -> Optional[Path]:
        if not self.json_files:
            return None
        exact = f"{self.base_name}_openpose".lower()
        for path in sorted(self.json_files, key=lambda p: p.name.lower()):
            if path.stem.lower() == exact:
                return path
        return sorted(self.json_files, key=lambda p: p.name.lower())[0]


def import_cv() -> Tuple[Any, Any, Optional[str]]:
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore

        return cv2, np, None
    except Exception as exc:  # pragma: no cover - depends on local environment
        return None, None, str(exc)


def resolve_default_openpose_root() -> Path:
    for candidate in DEFAULT_OPENPOSE_ROOTS:
        if candidate and candidate.exists():
            return candidate.absolute()
    return (PLUGIN_ROOT.parent.parent / "models" / "openpose").absolute()


def strip_pose_file_suffix(stem: str) -> str:
    base = stem.strip()
    while True:
        match = POSE_FILE_SUFFIX_RE.search(base)
        if not match:
            return base.strip()
        base = base[: match.start()].rstrip()


def classify_asset(path: Path) -> Optional[str]:
    suffix = path.suffix.lower()
    stem = path.stem.lower()
    if suffix == ".png":
        if stem.endswith("_bone_structure"):
            return "bone"
        if stem.endswith("_bone_structure_full"):
            return "bone_full"
        if re.search(r"_depth(_[a-z0-9]+)?$", stem, re.IGNORECASE):
            return "depth"
        return None
    if suffix == ".json":
        if stem.endswith("_openpose"):
            return "json"
        return None
    return None


def iter_asset_groups(root: Path) -> List[AssetGroup]:
    groups: Dict[Tuple[str, str], AssetGroup] = {}

    for path in sorted(root.rglob("*"), key=lambda p: str(p).lower()):
        if not path.is_file():
            continue
        kind = classify_asset(path)
        if kind is None:
            continue
        base = strip_pose_file_suffix(path.stem)
        if not base:
            continue
        key = (str(path.parent.resolve()).lower(), base.lower())
        if key not in groups:
            groups[key] = AssetGroup(folder=path.parent, base_name=base)
        group = groups[key]
        group.files.append(path)
        if kind == "depth":
            group.depth_files.append(path)
        elif kind == "bone":
            group.bone_files.append(path)
        elif kind == "bone_full":
            group.bone_full_files.append(path)
        elif kind == "json":
            group.json_files.append(path)

    return [groups[key] for key in sorted(groups.keys())]


def read_png_size(path: Path) -> Optional[ImageInfo]:
    try:
        with path.open("rb") as handle:
            header = handle.read(33)
        if len(header) < 33 or not header.startswith(b"\x89PNG\r\n\x1a\n"):
            return None
        width, height = struct.unpack(">II", header[16:24])
        color_type = header[25]
        channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}.get(color_type, 0)
        return ImageInfo(width=int(width), height=int(height), channels=channels)
    except Exception:
        return None


def read_image_info(path: Optional[Path], cv2: Any = None) -> Optional[ImageInfo]:
    if path is None:
        return None
    if cv2 is not None:
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is not None:
            height, width = image.shape[:2]
            channels = 1 if len(image.shape) == 2 else int(image.shape[2])
            return ImageInfo(width=int(width), height=int(height), channels=channels)
    return read_png_size(path)


def load_json_info(path: Optional[Path]) -> Optional[JsonInfo]:
    if path is None:
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        return JsonInfo(
            path=path,
            canvas_width=None,
            canvas_height=None,
            body_point_count=0,
            valid_body_count=0,
            keypoints=[],
            error=str(exc),
        )

    keypoints: List[float] = []
    canvas_width = data.get("canvas_width", data.get("width")) if isinstance(data, dict) else None
    canvas_height = data.get("canvas_height", data.get("height")) if isinstance(data, dict) else None

    if isinstance(data, dict):
        people = data.get("people")
        if isinstance(people, list) and people and isinstance(people[0], dict):
            raw = people[0].get("pose_keypoints_2d") or people[0].get("keypoints") or []
            if isinstance(raw, list):
                keypoints = [float(value) for value in raw if isinstance(value, (int, float))]
        elif isinstance(data.get("pose_keypoints_2d"), list):
            keypoints = [float(value) for value in data["pose_keypoints_2d"] if isinstance(value, (int, float))]
        elif isinstance(data.get("keypoints"), list):
            keypoints = [float(value) for value in data["keypoints"] if isinstance(value, (int, float))]
    elif isinstance(data, list) and all(isinstance(value, (int, float)) for value in data):
        keypoints = [float(value) for value in data]

    point_count = len(keypoints) // 3
    valid_count = 0
    for index in range(point_count):
        confidence = keypoints[index * 3 + 2]
        if confidence > 0:
            valid_count += 1

    return JsonInfo(
        path=path,
        canvas_width=int(canvas_width) if isinstance(canvas_width, (int, float)) else None,
        canvas_height=int(canvas_height) if isinstance(canvas_height, (int, float)) else None,
        body_point_count=point_count,
        valid_body_count=valid_count,
        keypoints=keypoints,
    )


def point_neighbors() -> Dict[int, List[int]]:
    neighbors: Dict[int, List[int]] = {index: [] for index in range(18)}
    for start, end in OPENPOSE_18_LIMBS:
        neighbors[start].append(end)
        neighbors[end].append(start)
    return neighbors


NEIGHBORS = point_neighbors()


def color_candidates_for_index(
    image_rgb: Any,
    encoded_rgb: Any,
    color: Tuple[int, int, int],
    color_code: int,
    index: int,
    cv2: Any,
    np: Any,
) -> List[CandidatePoint]:
    exact_mask = encoded_rgb == color_code
    if exact_mask.any():
        mask = exact_mask.astype(np.uint8)
    else:
        target = np.array(color, dtype=np.int16)
        diff = np.max(np.abs(image_rgb.astype(np.int16) - target), axis=2)
        mask = (diff <= 2).astype(np.uint8)
    component_count, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    candidates: List[CandidatePoint] = []

    for component_index in range(1, component_count):
        x, y, width, height, area = [int(value) for value in stats[component_index]]
        if area < 5 or area > 240:
            continue
        if width > 32 or height > 32:
            continue
        if min(width, height) <= 1:
            continue
        area_score = 1.0 - min(1.0, abs(area - 49) / 80.0)
        square_score = 1.0 - min(1.0, abs(width - height) / 18.0)
        size_score = 1.0 - min(1.0, (abs(width - 9) + abs(height - 9)) / 40.0)
        score = area_score * 0.45 + square_score * 0.35 + size_score * 0.20
        candidates.append(
            CandidatePoint(
                index=index,
                x=float(centroids[component_index][0]),
                y=float(centroids[component_index][1]),
                area=area,
                width=width,
                height=height,
                score=score,
            )
        )

    candidates.sort(key=lambda item: item.score, reverse=True)
    return candidates[:8]


def choose_candidate(
    index: int,
    candidates: Sequence[CandidatePoint],
    selected: Sequence[Optional[CandidatePoint]],
    diagonal: float,
) -> CandidatePoint:
    if len(candidates) == 1:
        return candidates[0]

    best_candidate = candidates[0]
    best_score = -float("inf")
    for candidate in candidates:
        neighbor_distances = []
        for neighbor_index in NEIGHBORS[index]:
            neighbor = selected[neighbor_index]
            if neighbor is None:
                continue
            distance = math.hypot(candidate.x - neighbor.x, candidate.y - neighbor.y)
            neighbor_distances.append(distance)

        graph_score = 0.0
        if neighbor_distances:
            average_distance = sum(neighbor_distances) / len(neighbor_distances)
            graph_score = 1.0 - min(1.0, average_distance / max(1.0, diagonal * 0.45))
        total = candidate.score * 0.45 + graph_score * 0.55
        if total > best_score:
            best_score = total
            best_candidate = candidate
    return best_candidate


def extract_bone_keypoints(path: Path, cv2: Any, np: Any) -> Optional[BoneExtraction]:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return None
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image_rgb.shape[:2]
    diagonal = math.hypot(width, height)
    encoded_rgb = (
        (image_rgb[:, :, 0].astype(np.uint32) << 16)
        | (image_rgb[:, :, 1].astype(np.uint32) << 8)
        | image_rgb[:, :, 2].astype(np.uint32)
    )

    candidates_by_index = [
        color_candidates_for_index(image_rgb, encoded_rgb, color, OPENPOSE_COLOR_CODES[index], index, cv2, np)
        for index, color in enumerate(OPENPOSE_COLORS_RGB)
    ]
    selected: List[Optional[CandidatePoint]] = [None] * 18

    for index, candidates in enumerate(candidates_by_index):
        if len(candidates) == 1:
            selected[index] = candidates[0]

    for _ in range(5):
        changed = False
        for index, candidates in enumerate(candidates_by_index):
            if not candidates:
                continue
            choice = choose_candidate(index, candidates, selected, diagonal)
            current = selected[index]
            if current is None or abs(current.x - choice.x) > 0.01 or abs(current.y - choice.y) > 0.01:
                selected[index] = choice
                changed = True
        if not changed:
            break

    keypoints: List[float] = []
    missing = []
    for index, point in enumerate(selected):
        if point is None:
            keypoints.extend([0.0, 0.0, 0.0])
            missing.append(index)
        else:
            keypoints.extend([float(point.x), float(point.y), 1.0])

    selected_count = 18 - len(missing)
    warnings: List[str] = []
    if selected_count < 12:
        warnings.append("bone_extract_low_confidence")
    if any(index in missing for index in [1, 2, 5, 8, 11]):
        warnings.append("bone_missing_core_torso_points")

    return BoneExtraction(
        path=path,
        width=int(width),
        height=int(height),
        keypoints=keypoints,
        selected_count=selected_count,
        candidate_counts=[len(candidates) for candidates in candidates_by_index],
        missing_indices=missing,
        warnings=warnings,
    )


def scaled_json_keypoints(info: JsonInfo, target_width: int, target_height: int) -> List[float]:
    keypoints = list(info.keypoints[: 18 * 3])
    if len(keypoints) < 18 * 3:
        keypoints.extend([0.0] * (18 * 3 - len(keypoints)))

    if info.canvas_width and info.canvas_height and info.canvas_width > 0 and info.canvas_height > 0:
        scale_x = target_width / info.canvas_width
        scale_y = target_height / info.canvas_height
    else:
        scale_x = scale_y = 1.0

    out = []
    for index in range(18):
        x, y, confidence = keypoints[index * 3 : index * 3 + 3]
        if confidence > 0:
            out.extend([float(x) * scale_x, float(y) * scale_y, float(confidence)])
        else:
            out.extend([0.0, 0.0, 0.0])
    return out


def compare_keypoints(a: Sequence[float], b: Sequence[float]) -> Dict[str, Any]:
    distances = []
    for index in range(18):
        ax, ay, ac = a[index * 3 : index * 3 + 3]
        bx, by, bc = b[index * 3 : index * 3 + 3]
        if ac <= 0 or bc <= 0:
            continue
        distances.append(math.hypot(ax - bx, ay - by))

    if not distances:
        return {"count": 0, "median_px": None, "mean_px": None, "max_px": None}

    distances_sorted = sorted(distances)
    midpoint = len(distances_sorted) // 2
    if len(distances_sorted) % 2:
        median = distances_sorted[midpoint]
    else:
        median = (distances_sorted[midpoint - 1] + distances_sorted[midpoint]) / 2.0
    return {
        "count": len(distances),
        "median_px": round(float(median), 3),
        "mean_px": round(float(sum(distances) / len(distances)), 3),
        "max_px": round(float(max(distances)), 3),
    }


def is_json_body_count_valid(info: Optional[JsonInfo], expected_body_points: int) -> bool:
    if info is None or info.error:
        return False
    return info.body_point_count == expected_body_points


def make_openpose_payload(group: AssetGroup, extraction: BoneExtraction) -> Dict[str, Any]:
    return {
        "version": 1.0,
        "canvas_width": extraction.width,
        "canvas_height": extraction.height,
        "people": [
            {
                "pose_keypoints_2d": extraction.keypoints,
                "face_keypoints_2d": [],
                "hand_left_keypoints_2d": [],
                "hand_right_keypoints_2d": [],
            }
        ],
        "meta": {
            "schema": "pal_pose_asset_audit/reconstructed_from_bone/v1",
            "source_file": str(extraction.path),
            "base_name": group.base_name,
            "selected_body_points": extraction.selected_count,
            "missing_body_indices": extraction.missing_indices,
        },
    }


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def relpath(path: Optional[Path], root: Path) -> str:
    if path is None:
        return ""
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except ValueError:
        return str(path)


def build_folder_json_index(groups: Sequence[AssetGroup]) -> Dict[str, List[JsonInfo]]:
    by_folder: Dict[str, List[JsonInfo]] = {}
    for group in groups:
        info = load_json_info(group.json_path)
        if info is None or info.error or len(info.keypoints) < 18 * 3:
            continue
        key = str(group.folder.resolve()).lower()
        by_folder.setdefault(key, []).append(info)
    return by_folder


def find_best_json_match(
    extraction: BoneExtraction,
    folder_jsons: Sequence[JsonInfo],
    own_json: Optional[JsonInfo],
) -> Dict[str, Any]:
    matches = []
    for info in folder_jsons:
        scaled = scaled_json_keypoints(info, extraction.width, extraction.height)
        comparison = compare_keypoints(extraction.keypoints, scaled)
        if comparison["count"] < 8 or comparison["median_px"] is None:
            continue
        matches.append((float(comparison["median_px"]), float(comparison["mean_px"]), comparison["count"], info.path))

    matches.sort(key=lambda item: (item[0], item[1], str(item[3]).lower()))
    best = matches[0] if matches else None

    own = None
    if own_json is not None:
        for match in matches:
            if match[3].resolve() == own_json.path.resolve():
                own = match
                break

    return {
        "best_json": str(best[3]) if best else "",
        "best_json_base": strip_pose_file_suffix(best[3].stem) if best else "",
        "best_median_px": round(best[0], 3) if best else None,
        "best_mean_px": round(best[1], 3) if best else None,
        "best_compared_points": best[2] if best else 0,
        "own_median_px": round(own[0], 3) if own else None,
        "own_mean_px": round(own[1], 3) if own else None,
        "own_compared_points": own[2] if own else 0,
    }


def audit_groups(args: argparse.Namespace) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    cv2, np, cv_error = import_cv()
    groups = iter_asset_groups(args.root)
    folder_json_index = build_folder_json_index(groups)

    summary = {
        "root": str(args.root),
        "groups": len(groups),
        "missing_depth": 0,
        "missing_bone": 0,
        "missing_json": 0,
        "invalid_json": 0,
        "image_dimension_mismatch": 0,
        "bone_extract_low_confidence": 0,
        "bone_json_mismatch": 0,
        "json_written": 0,
        "cv_available": cv2 is not None,
        "cv_error": cv_error or "",
    }
    rows: List[Dict[str, Any]] = []

    for group in groups:
        issues: List[str] = []
        depth = group.depth
        bone = group.bone
        json_path = group.json_path

        if depth is None:
            summary["missing_depth"] += 1
            issues.append("missing_depth")
        if bone is None:
            summary["missing_bone"] += 1
            issues.append("missing_bone_structure")
        if json_path is None:
            summary["missing_json"] += 1
            issues.append("missing_openpose_json")

        depth_info = read_image_info(depth, cv2)
        bone_info = read_image_info(bone, cv2)
        json_info = load_json_info(json_path)

        if json_info is not None:
            if json_info.error:
                summary["invalid_json"] += 1
                issues.append("invalid_json_parse")
            elif json_info.body_point_count != args.expected_body_points:
                summary["invalid_json"] += 1
                issues.append(f"invalid_json_body_points:{json_info.body_point_count}")

        dimensions = []
        if depth_info:
            dimensions.append(("depth", depth_info.width, depth_info.height))
        if bone_info:
            dimensions.append(("bone", bone_info.width, bone_info.height))
        if json_info and json_info.canvas_width and json_info.canvas_height:
            dimensions.append(("json", json_info.canvas_width, json_info.canvas_height))
        unique_dimensions = {(width, height) for _kind, width, height in dimensions}
        if len(unique_dimensions) > 1:
            summary["image_dimension_mismatch"] += 1
            issues.append("dimension_mismatch")

        extraction = None
        match_info: Dict[str, Any] = {}
        if bone is not None and cv2 is not None and np is not None:
            extraction = extract_bone_keypoints(bone, cv2, np)
            if extraction is None:
                issues.append("bone_extract_failed")
            else:
                for warning in extraction.warnings:
                    if warning not in issues:
                        issues.append(warning)
                    if warning == "bone_extract_low_confidence":
                        summary["bone_extract_low_confidence"] += 1

                folder_key = str(group.folder.resolve()).lower()
                match_info = find_best_json_match(
                    extraction,
                    folder_json_index.get(folder_key, []),
                    json_info,
                )
                own_median = match_info.get("own_median_px")
                best_median = match_info.get("best_median_px")
                best_base = match_info.get("best_json_base") or ""
                if (
                    is_json_body_count_valid(json_info, args.expected_body_points)
                    and extraction.selected_count >= args.min_bone_points
                    and own_median is not None
                    and best_median is not None
                    and (
                        own_median > args.mismatch_median_px
                        or (best_base and best_base != group.base_name and own_median - best_median > args.mismatch_margin_px)
                    )
                ):
                    summary["bone_json_mismatch"] += 1
                    issues.append("bone_json_mismatch")

                if json_path is None and args.write_missing_json and extraction.selected_count >= args.min_bone_points:
                    target = group.folder / f"{group.base_name}_openpose.json"
                    payload = make_openpose_payload(group, extraction)
                    write_json(target, payload)
                    summary["json_written"] += 1
                    json_path = target
                    json_info = load_json_info(json_path)
                    issues = [issue for issue in issues if issue != "missing_openpose_json"]

        elif bone is not None and cv2 is None:
            issues.append("cv_unavailable_bone_not_checked")

        row = {
            "base_name": group.base_name,
            "folder": relpath(group.folder, args.root),
            "depth": relpath(depth, args.root),
            "bone_structure": relpath(bone, args.root),
            "openpose_json": relpath(json_path, args.root),
            "depth_dimensions": f"{depth_info.width}x{depth_info.height}" if depth_info else "",
            "bone_dimensions": f"{bone_info.width}x{bone_info.height}" if bone_info else "",
            "json_dimensions": (
                f"{json_info.canvas_width}x{json_info.canvas_height}"
                if json_info and json_info.canvas_width and json_info.canvas_height
                else ""
            ),
            "json_body_points": json_info.body_point_count if json_info else "",
            "json_valid_points": json_info.valid_body_count if json_info else "",
            "bone_extracted_points": extraction.selected_count if extraction else "",
            "bone_missing_indices": extraction.missing_indices if extraction else [],
            "best_json_base": match_info.get("best_json_base", ""),
            "best_json": relpath(Path(match_info["best_json"]), args.root) if match_info.get("best_json") else "",
            "best_median_px": match_info.get("best_median_px"),
            "own_median_px": match_info.get("own_median_px"),
            "issues": issues,
        }
        if issues or not args.only_issues:
            rows.append(row)

    return summary, rows


def write_csv_report(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "base_name",
        "folder",
        "issues",
        "depth",
        "bone_structure",
        "openpose_json",
        "depth_dimensions",
        "bone_dimensions",
        "json_dimensions",
        "json_body_points",
        "json_valid_points",
        "bone_extracted_points",
        "bone_missing_indices",
        "best_json_base",
        "best_json",
        "best_median_px",
        "own_median_px",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            output = dict(row)
            output["issues"] = ";".join(row.get("issues", []))
            output["bone_missing_indices"] = ",".join(str(value) for value in row.get("bone_missing_indices", []))
            writer.writerow({field: output.get(field, "") for field in fieldnames})


def uri_for_report(root: Path, value: str) -> str:
    if not value:
        return ""
    path = Path(value)
    if not path.is_absolute():
        path = root / value
    try:
        return path.resolve().as_uri()
    except Exception:
        return ""


def write_html_report(path: Path, root: Path, summary: Dict[str, Any], rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    parts = [
        "<!doctype html>",
        "<html>",
        "<head>",
        "<meta charset=\"utf-8\">",
        "<title>Pose Asset Audit</title>",
        "<style>",
        "body{font-family:Segoe UI,Arial,sans-serif;margin:24px;background:#f7f7f5;color:#1f2328}",
        "table{border-collapse:collapse;width:100%;background:white}",
        "th,td{border:1px solid #d8dee4;padding:6px 8px;vertical-align:top;font-size:13px}",
        "th{position:sticky;top:0;background:#eef1f4;text-align:left}",
        "img{max-width:180px;max-height:140px;background:#111;object-fit:contain}",
        ".issues{font-weight:600;color:#9a3412}",
        ".muted{color:#667085}",
        ".summary{display:flex;flex-wrap:wrap;gap:8px;margin:12px 0 18px}",
        ".pill{background:white;border:1px solid #d8dee4;border-radius:6px;padding:6px 8px}",
        "</style>",
        "</head>",
        "<body>",
        "<h1>Pose Asset Audit</h1>",
        f"<div class=\"muted\">Root: {html.escape(str(root))}</div>",
        "<div class=\"summary\">",
    ]
    for key in [
        "groups",
        "missing_json",
        "invalid_json",
        "missing_depth",
        "missing_bone",
        "image_dimension_mismatch",
        "bone_extract_low_confidence",
        "bone_json_mismatch",
        "json_written",
    ]:
        parts.append(f"<div class=\"pill\">{html.escape(key)}: {html.escape(str(summary.get(key, '')))}</div>")
    parts.extend(
        [
            "</div>",
            "<table>",
            "<thead><tr>",
            "<th>Base</th><th>Issues</th><th>Depth</th><th>Bone</th><th>JSON / Match</th><th>Details</th>",
            "</tr></thead><tbody>",
        ]
    )

    for row in rows:
        depth_uri = uri_for_report(root, str(row.get("depth", "")))
        bone_uri = uri_for_report(root, str(row.get("bone_structure", "")))
        issues = "; ".join(row.get("issues", []))
        details = [
            f"depth {row.get('depth_dimensions') or '-'}",
            f"bone {row.get('bone_dimensions') or '-'}",
            f"json {row.get('json_dimensions') or '-'}",
            f"json points {row.get('json_body_points') or '-'}",
            f"bone points {row.get('bone_extracted_points') or '-'}",
        ]
        if row.get("best_json_base"):
            details.append(
                f"best {row.get('best_json_base')} median {row.get('best_median_px')} px; "
                f"own {row.get('own_median_px')} px"
            )

        parts.extend(
            [
                "<tr>",
                f"<td><strong>{html.escape(str(row.get('base_name', '')))}</strong><br>"
                f"<span class=\"muted\">{html.escape(str(row.get('folder', '')))}</span></td>",
                f"<td class=\"issues\">{html.escape(issues)}</td>",
                f"<td>{f'<img src=\"{html.escape(depth_uri)}\"><br>' if depth_uri else ''}"
                f"<span class=\"muted\">{html.escape(str(row.get('depth', '')))}</span></td>",
                f"<td>{f'<img src=\"{html.escape(bone_uri)}\"><br>' if bone_uri else ''}"
                f"<span class=\"muted\">{html.escape(str(row.get('bone_structure', '')))}</span></td>",
                f"<td>{html.escape(str(row.get('openpose_json', '')))}<br>"
                f"<span class=\"muted\">{html.escape(str(row.get('best_json', '')))}</span></td>",
                f"<td>{html.escape(' | '.join(details))}</td>",
                "</tr>",
            ]
        )

    parts.extend(["</tbody></table>", "</body>", "</html>"])
    path.write_text("\n".join(parts), encoding="utf-8")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit OpenPose depth, bone_structure, and JSON asset groups.")
    parser.add_argument("--root", type=Path, default=resolve_default_openpose_root(), help="OpenPose dataset root.")
    parser.add_argument("--scope", default="", help="Optional subfolder below --root, e.g. sitting/F/nsfw/sitting.")
    parser.add_argument("--report", type=Path, default=PLUGIN_ROOT / "pose_asset_audit_report.json")
    parser.add_argument("--csv-report", type=Path, default=PLUGIN_ROOT / "pose_asset_audit_report.csv")
    parser.add_argument("--html-report", type=Path, default=PLUGIN_ROOT / "pose_asset_audit_report.html")
    parser.add_argument("--only-issues", action="store_true", help="Write only rows with at least one issue.")
    parser.add_argument("--write-missing-json", action="store_true", help="Create missing *_openpose.json files from bone_structure PNGs.")
    parser.add_argument("--expected-body-points", type=int, default=18, help="Expected body keypoint count in *_openpose.json.")
    parser.add_argument("--min-bone-points", type=int, default=12, help="Minimum extracted bone points needed for matching/writing.")
    parser.add_argument("--mismatch-median-px", type=float, default=24.0, help="Own JSON median distance threshold.")
    parser.add_argument("--mismatch-margin-px", type=float, default=12.0, help="Best-other JSON margin threshold.")
    args = parser.parse_args(argv)
    args.root = args.root.expanduser()
    if args.scope:
        args.root = (args.root / args.scope).absolute()
    else:
        args.root = args.root.expanduser().absolute()
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if not args.root.exists():
        print(f"OpenPose root does not exist: {args.root}", file=sys.stderr)
        return 2

    summary, rows = audit_groups(args)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_csv_report(args.csv_report, rows)
    write_html_report(args.html_report, args.root, summary, rows)

    print(f"Audited {summary['groups']} asset groups")
    print(f"Report: {args.report}")
    print(f"CSV: {args.csv_report}")
    print(f"HTML: {args.html_report}")
    for key in [
        "missing_json",
        "json_written",
        "invalid_json",
        "missing_depth",
        "missing_bone",
        "image_dimension_mismatch",
        "bone_extract_low_confidence",
        "bone_json_mismatch",
    ]:
        print(f"{key}: {summary[key]}")
    if not summary["cv_available"]:
        print(f"WARNING: OpenCV/Numpy unavailable; bone image checks skipped: {summary['cv_error']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
