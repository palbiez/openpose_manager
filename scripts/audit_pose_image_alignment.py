#!/usr/bin/env python3
"""Audit OpenPose bone/JSON alignment against rendered depth/normal images.

This audit is intentionally conservative.  It projects OpenPose JSON limbs into
the rendered image space, scores the normal orientation and a horizontal flip
against a foreground mask, and reports likely mirrored or mismatched assets.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageFilter


PLUGIN_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OPENPOSE_ROOTS = [
    Path(os.environ["OPENPOSE_MODELS_PATH"]) if os.environ.get("OPENPOSE_MODELS_PATH") else None,
    PLUGIN_ROOT.parent.parent / "models" / "openpose",
    Path.cwd() / "models" / "openpose",
    Path(r"C:\EasyDiffusion\stable-diffusion\stable-diffusion-webui\models\openpose"),
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


@dataclass
class AssetGroup:
    folder: Path
    base_name: str
    files: List[Path] = field(default_factory=list)
    depth_files: List[Path] = field(default_factory=list)
    normal_files: List[Path] = field(default_factory=list)
    lineart_files: List[Path] = field(default_factory=list)
    bone_files: List[Path] = field(default_factory=list)
    bone_full_files: List[Path] = field(default_factory=list)
    json_files: List[Path] = field(default_factory=list)

    @property
    def depth(self) -> Optional[Path]:
        return first_sorted(self.depth_files)

    @property
    def normal(self) -> Optional[Path]:
        return first_sorted(self.normal_files)

    @property
    def lineart(self) -> Optional[Path]:
        return first_sorted(self.lineart_files)

    @property
    def bone(self) -> Optional[Path]:
        return first_sorted(self.bone_files) or first_sorted(self.bone_full_files)

    @property
    def json_path(self) -> Optional[Path]:
        if not self.json_files:
            return None
        exact = f"{self.base_name}_openpose".lower()
        for path in sorted(self.json_files, key=lambda item: item.name.lower()):
            if path.stem.lower() == exact:
                return path
        return first_sorted(self.json_files)


@dataclass
class PoseJson:
    canvas_width: Optional[float]
    canvas_height: Optional[float]
    keypoints: List[float]
    valid_points: int
    error: str = ""


@dataclass
class MaskInfo:
    source_kind: str
    source_path: Path
    original_width: int
    original_height: int
    scale: float
    mask: np.ndarray


def first_sorted(paths: Sequence[Path]) -> Optional[Path]:
    return sorted(paths, key=lambda item: item.name.lower())[0] if paths else None


def resolve_default_openpose_root() -> Path:
    for candidate in DEFAULT_OPENPOSE_ROOTS:
        if candidate and candidate.exists():
            return candidate.absolute()
    fallback = DEFAULT_OPENPOSE_ROOTS[-1]
    assert fallback is not None
    return fallback.absolute()


def strip_pose_file_suffix(stem: str) -> str:
    base = stem.strip()
    while True:
        match = POSE_FILE_SUFFIX_RE.search(base)
        if match is None:
            return base
        base = base[: match.start()].rstrip()


def classify_asset(path: Path) -> Optional[str]:
    stem = path.stem.lower()
    suffix = path.suffix.lower()
    if suffix == ".png":
        if stem.endswith("_bone_structure"):
            return "bone"
        if stem.endswith("_bone_structure_full"):
            return "bone_full"
        if re.search(r"_depth(_[a-z0-9]+)?$", stem, re.IGNORECASE):
            return "depth"
        if stem.endswith("_normal"):
            return "normal"
        if stem.endswith("_lineart") or stem.endswith("_line_art") or stem.endswith("_linart"):
            return "lineart"
    if suffix == ".json" and stem.endswith("_openpose"):
        return "json"
    return None


def iter_asset_groups(root: Path) -> List[AssetGroup]:
    groups: Dict[Tuple[str, str], AssetGroup] = {}
    for path in sorted(root.rglob("*"), key=lambda item: str(item).lower()):
        if not path.is_file():
            continue
        kind = classify_asset(path)
        if kind is None:
            continue
        base_name = strip_pose_file_suffix(path.stem)
        key = (str(path.parent.resolve()).lower(), base_name.lower())
        group = groups.setdefault(key, AssetGroup(folder=path.parent, base_name=base_name))
        group.files.append(path)
        if kind == "depth":
            group.depth_files.append(path)
        elif kind == "normal":
            group.normal_files.append(path)
        elif kind == "lineart":
            group.lineart_files.append(path)
        elif kind == "bone":
            group.bone_files.append(path)
        elif kind == "bone_full":
            group.bone_full_files.append(path)
        elif kind == "json":
            group.json_files.append(path)
    return [groups[key] for key in sorted(groups.keys())]


def relpath(path: Optional[Path], root: Path) -> str:
    if path is None:
        return ""
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except ValueError:
        return str(path)


def load_pose_json(path: Optional[Path]) -> Optional[PoseJson]:
    if path is None:
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        return PoseJson(None, None, [], 0, str(exc))

    if not isinstance(payload, dict):
        return PoseJson(None, None, [], 0, "unsupported_json_shape")

    raw: Any = []
    people = payload.get("people")
    if isinstance(people, list) and people and isinstance(people[0], dict):
        raw = people[0].get("pose_keypoints_2d") or people[0].get("keypoints") or []
    elif isinstance(payload.get("pose_keypoints_2d"), list):
        raw = payload.get("pose_keypoints_2d")
    elif isinstance(payload.get("keypoints"), list):
        raw = payload.get("keypoints")

    keypoints = [float(value) for value in raw if isinstance(value, (int, float))]
    body_points = min(18, len(keypoints) // 3)
    keypoints = keypoints[: body_points * 3]
    keypoints.extend([0.0] * (18 * 3 - len(keypoints)))

    canvas_width = payload.get("canvas_width", payload.get("width"))
    canvas_height = payload.get("canvas_height", payload.get("height"))
    width = float(canvas_width) if isinstance(canvas_width, (int, float)) and canvas_width > 0 else None
    height = float(canvas_height) if isinstance(canvas_height, (int, float)) and canvas_height > 0 else None
    valid_points = sum(1 for index in range(18) if keypoints[index * 3 + 2] > 0)
    return PoseJson(width, height, keypoints, valid_points)


def resized_image(path: Path, max_side: int, mode: str) -> Tuple[Image.Image, Tuple[int, int], float]:
    image = Image.open(path)
    original_size = image.size
    scale = min(1.0, max_side / max(original_size))
    target_size = (
        max(1, int(round(original_size[0] * scale))),
        max(1, int(round(original_size[1] * scale))),
    )
    if image.mode != mode:
        image = image.convert(mode)
    if target_size != original_size:
        image = image.resize(target_size, Image.Resampling.BILINEAR)
    return image, original_size, scale


def normal_mask(path: Path, max_side: int) -> MaskInfo:
    image, original_size, scale = resized_image(path, max_side, "RGB")
    rgb = np.asarray(image).astype(np.float32) / 255.0
    brightness = rgb.max(axis=2)
    saturation = rgb.max(axis=2) - rgb.min(axis=2)
    mask = (brightness > 0.08) & ((saturation > 0.025) | (brightness > 0.16))
    return MaskInfo("normal", path, original_size[0], original_size[1], scale, mask)


def depth_mask(path: Path, max_side: int) -> MaskInfo:
    image, original_size, scale = resized_image(path, max_side, "F")
    gray = np.asarray(image).astype(np.float32)
    max_value = float(gray.max())
    if max_value > 0:
        gray = gray / max_value

    row_background = np.percentile(gray, 8, axis=1)
    diff = gray - row_background[:, None]
    nonzero = gray[gray > 0.01]
    high_reference = float(np.percentile(nonzero, 65)) if nonzero.size else 0.2
    bright = gray > max(0.16, high_reference * 0.55)

    # Row-background subtraction removes broad depth-floor gradients while
    # preserving the person silhouette in white-on-black depth maps.
    mask = (diff > 0.055) | (bright & (diff > 0.025))
    return MaskInfo("depth", path, original_size[0], original_size[1], scale, mask)


def make_foreground_mask(group: AssetGroup, max_side: int) -> Optional[MaskInfo]:
    if group.normal is not None:
        return normal_mask(group.normal, max_side)
    if group.depth is not None:
        return depth_mask(group.depth, max_side)
    return None


def dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask
    size = radius * 2 + 1
    image = Image.fromarray(mask.astype(np.uint8) * 255, "L")
    return np.asarray(image.filter(ImageFilter.MaxFilter(size))) > 0


def transformed_point(
    pose: PoseJson,
    index: int,
    mask_info: MaskInfo,
    flip_x: bool,
) -> Optional[Tuple[float, float]]:
    x, y, confidence = pose.keypoints[index * 3 : index * 3 + 3]
    if confidence <= 0:
        return None

    canvas_width = pose.canvas_width or float(mask_info.original_width)
    canvas_height = pose.canvas_height or float(mask_info.original_height)
    x_scaled = x * (mask_info.original_width * mask_info.scale) / canvas_width
    y_scaled = y * (mask_info.original_height * mask_info.scale) / canvas_height
    if flip_x:
        x_scaled = mask_info.original_width * mask_info.scale - 1.0 - x_scaled
    return x_scaled, y_scaled


def sampled_pose_points(pose: PoseJson, mask_info: MaskInfo, flip_x: bool) -> List[Tuple[float, float]]:
    points: List[Tuple[float, float]] = []
    for index in range(18):
        point = transformed_point(pose, index, mask_info, flip_x)
        if point is not None:
            points.append(point)

    for start, end in OPENPOSE_18_LIMBS:
        start_point = transformed_point(pose, start, mask_info, flip_x)
        end_point = transformed_point(pose, end, mask_info, flip_x)
        if start_point is None or end_point is None:
            continue
        for t in (0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875):
            points.append(
                (
                    start_point[0] * (1.0 - t) + end_point[0] * t,
                    start_point[1] * (1.0 - t) + end_point[1] * t,
                )
            )
    return points


def score_points(mask: np.ndarray, points: Sequence[Tuple[float, float]]) -> Tuple[float, int, int]:
    height, width = mask.shape
    if not points:
        return 0.0, 0, 0
    hits = 0
    in_bounds = 0
    for x_value, y_value in points:
        x = int(round(x_value))
        y = int(round(y_value))
        if 0 <= x < width and 0 <= y < height:
            in_bounds += 1
            if mask[y, x]:
                hits += 1
    return hits / len(points), hits, in_bounds


def classify_alignment(
    original_score: float,
    flipped_score: float,
    args: argparse.Namespace,
) -> str:
    delta = flipped_score - original_score
    if flipped_score >= args.mirror_min_score and delta >= args.mirror_min_delta:
        return "mirror_candidate"
    if max(original_score, flipped_score) < args.mismatch_max_score:
        return "mismatch_candidate"
    if original_score >= args.ok_min_score and delta < args.mirror_min_delta:
        return "ok"
    return "ambiguous"


def audit_group(group: AssetGroup, root: Path, args: argparse.Namespace) -> Dict[str, Any]:
    issues: List[str] = []
    pose_json = load_pose_json(group.json_path)
    if group.depth is None:
        issues.append("missing_depth")
    if group.bone is None:
        issues.append("missing_bone_structure")
    if group.json_path is None:
        issues.append("missing_openpose_json")
    if pose_json is not None and pose_json.error:
        issues.append("invalid_openpose_json")

    mask_info: Optional[MaskInfo] = None
    original_score: Optional[float] = None
    flipped_score: Optional[float] = None
    original_hits: Optional[int] = None
    flipped_hits: Optional[int] = None
    sample_count: Optional[int] = None
    classification = "not_scored"

    if not issues or (pose_json is not None and group.depth is not None and group.bone is not None):
        if pose_json is not None and not pose_json.error and pose_json.valid_points >= args.min_valid_points:
            mask_info = make_foreground_mask(group, args.max_side)
            if mask_info is not None:
                radius = max(args.min_dilate_px, int(round(min(mask_info.mask.shape) / args.dilate_divisor)))
                scored_mask = dilate_mask(mask_info.mask, radius)
                original_points = sampled_pose_points(pose_json, mask_info, flip_x=False)
                flipped_points = sampled_pose_points(pose_json, mask_info, flip_x=True)
                original_score, original_hits, _ = score_points(scored_mask, original_points)
                flipped_score, flipped_hits, _ = score_points(scored_mask, flipped_points)
                sample_count = len(original_points)
                classification = classify_alignment(original_score, flipped_score, args)
                if classification != "ok":
                    issues.append(classification)
            else:
                issues.append("mask_unavailable")
        elif pose_json is None:
            issues.append("openpose_json_unreadable")
        elif pose_json.valid_points < args.min_valid_points:
            issues.append("openpose_json_low_valid_points")

    row: Dict[str, Any] = {
        "classification": classification,
        "issues": issues,
        "base_name": group.base_name,
        "folder": relpath(group.folder, root),
        "depth": relpath(group.depth, root),
        "normal": relpath(group.normal, root),
        "lineart": relpath(group.lineart, root),
        "bone_structure": relpath(group.bone, root),
        "openpose_json": relpath(group.json_path, root),
        "mask_source": mask_info.source_kind if mask_info else "",
        "mask_source_file": relpath(mask_info.source_path, root) if mask_info else "",
        "render_dimensions": f"{mask_info.original_width}x{mask_info.original_height}" if mask_info else "",
        "json_dimensions": (
            f"{int(pose_json.canvas_width)}x{int(pose_json.canvas_height)}"
            if pose_json and pose_json.canvas_width and pose_json.canvas_height
            else ""
        ),
        "json_valid_points": pose_json.valid_points if pose_json else "",
        "sample_count": sample_count if sample_count is not None else "",
        "original_score": round(original_score, 4) if original_score is not None else "",
        "flipped_score": round(flipped_score, 4) if flipped_score is not None else "",
        "flip_delta": round(flipped_score - original_score, 4)
        if original_score is not None and flipped_score is not None
        else "",
        "original_hits": original_hits if original_hits is not None else "",
        "flipped_hits": flipped_hits if flipped_hits is not None else "",
    }
    return row


def audit(args: argparse.Namespace) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    root = args.root.resolve()
    groups = iter_asset_groups(root)
    rows: List[Dict[str, Any]] = []
    summary = {
        "root": str(root),
        "groups": len(groups),
        "reported_rows": 0,
        "ok": 0,
        "mirror_candidate": 0,
        "mismatch_candidate": 0,
        "ambiguous": 0,
        "not_scored": 0,
        "missing_depth": 0,
        "missing_bone_structure": 0,
        "missing_openpose_json": 0,
    }

    for index, group in enumerate(groups, start=1):
        row = audit_group(group, root, args)
        classification = str(row["classification"])
        summary[classification] = summary.get(classification, 0) + 1
        for issue in row["issues"]:
            if issue.startswith("missing_") and issue in summary:
                summary[issue] += 1

        if not args.only_issues or row["issues"]:
            rows.append(row)

        if args.progress and (index % args.progress == 0 or index == len(groups)):
            print(f"scanned {index}/{len(groups)} groups", file=sys.stderr)

    summary["reported_rows"] = len(rows)
    return summary, rows


def write_json_report(path: Path, summary: Dict[str, Any], rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"summary": summary, "rows": rows}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv_report(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "classification",
        "issues",
        "base_name",
        "folder",
        "original_score",
        "flipped_score",
        "flip_delta",
        "mask_source",
        "render_dimensions",
        "json_dimensions",
        "json_valid_points",
        "sample_count",
        "depth",
        "normal",
        "lineart",
        "bone_structure",
        "openpose_json",
        "mask_source_file",
        "original_hits",
        "flipped_hits",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            output = dict(row)
            output["issues"] = ";".join(str(issue) for issue in row.get("issues", []))
            writer.writerow({field: output.get(field, "") for field in fieldnames})


def image_uri(root: Path, value: str) -> str:
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
        "<title>Pose Image Alignment Audit</title>",
        "<style>",
        "body{font-family:Segoe UI,Arial,sans-serif;margin:24px;background:#f7f7f5;color:#1f2328}",
        "table{border-collapse:collapse;width:100%;background:#fff}",
        "th,td{border:1px solid #ddd;padding:8px;vertical-align:top;font-size:12px}",
        "th{background:#eee;text-align:left;position:sticky;top:0}",
        "img{max-width:180px;max-height:220px;background:#111}",
        ".muted{color:#666;font-size:11px;word-break:break-all}",
        ".issue{font-weight:600;color:#9a3412}",
        ".summary{display:flex;gap:12px;flex-wrap:wrap;margin:12px 0 18px}",
        ".pill{background:#fff;border:1px solid #ddd;border-radius:6px;padding:8px 10px}",
        "</style>",
        "</head>",
        "<body>",
        "<h1>Pose Image Alignment Audit</h1>",
        f"<p class=\"muted\">Root: {html.escape(str(root))}</p>",
        "<div class=\"summary\">",
    ]
    for key in [
        "groups",
        "reported_rows",
        "ok",
        "mirror_candidate",
        "mismatch_candidate",
        "ambiguous",
        "not_scored",
        "missing_depth",
        "missing_bone_structure",
        "missing_openpose_json",
    ]:
        parts.append(f"<div class=\"pill\"><b>{html.escape(key)}</b><br>{html.escape(str(summary.get(key, 0)))}</div>")
    parts.extend(
        [
            "</div>",
            "<table>",
            "<thead><tr>",
            "<th>Class</th><th>Base</th><th>Scores</th><th>Bone</th><th>Render Mask</th><th>Files</th>",
            "</tr></thead><tbody>",
        ]
    )
    for row in rows:
        bone_uri = image_uri(root, str(row.get("bone_structure", "")))
        render_uri = image_uri(root, str(row.get("mask_source_file", "")) or str(row.get("depth", "")))
        issues = "; ".join(str(issue) for issue in row.get("issues", []))
        parts.extend(
            [
                "<tr>",
                f"<td><span class=\"issue\">{html.escape(str(row.get('classification', '')))}</span><br>"
                f"<span class=\"muted\">{html.escape(issues)}</span></td>",
                f"<td>{html.escape(str(row.get('base_name', '')))}<br>"
                f"<span class=\"muted\">{html.escape(str(row.get('folder', '')))}</span></td>",
                f"<td>orig: {html.escape(str(row.get('original_score', '')))}<br>"
                f"flip: {html.escape(str(row.get('flipped_score', '')))}<br>"
                f"delta: {html.escape(str(row.get('flip_delta', '')))}<br>"
                f"<span class=\"muted\">json {html.escape(str(row.get('json_dimensions', '')))}, "
                f"render {html.escape(str(row.get('render_dimensions', '')))}</span></td>",
                f"<td>{f'<img src=\"{html.escape(bone_uri)}\"><br>' if bone_uri else ''}"
                f"<span class=\"muted\">{html.escape(str(row.get('bone_structure', '')))}</span></td>",
                f"<td>{f'<img src=\"{html.escape(render_uri)}\"><br>' if render_uri else ''}"
                f"<span class=\"muted\">{html.escape(str(row.get('mask_source_file') or row.get('depth') or ''))}</span></td>",
                f"<td><span class=\"muted\">depth: {html.escape(str(row.get('depth', '')))}<br>"
                f"normal: {html.escape(str(row.get('normal', '')))}<br>"
                f"lineart: {html.escape(str(row.get('lineart', '')))}<br>"
                f"json: {html.escape(str(row.get('openpose_json', '')))}</span></td>",
                "</tr>",
            ]
        )
    parts.extend(["</tbody></table>", "</body>", "</html>"])
    path.write_text("\n".join(parts), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=resolve_default_openpose_root(), help="OpenPose model root.")
    parser.add_argument("--report", type=Path, default=PLUGIN_ROOT / "pose_image_alignment_audit.json")
    parser.add_argument("--csv-report", type=Path, default=PLUGIN_ROOT / "pose_image_alignment_audit.csv")
    parser.add_argument("--html-report", type=Path, default=PLUGIN_ROOT / "pose_image_alignment_audit.html")
    parser.add_argument("--only-issues", action="store_true", help="Only write rows that have issues.")
    parser.add_argument("--max-side", type=int, default=384, help="Resize render masks to this maximum side for scoring.")
    parser.add_argument("--min-valid-points", type=int, default=8)
    parser.add_argument("--min-dilate-px", type=int, default=8)
    parser.add_argument("--dilate-divisor", type=float, default=25.0)
    parser.add_argument("--mirror-min-score", type=float, default=0.72)
    parser.add_argument("--mirror-min-delta", type=float, default=0.14)
    parser.add_argument("--mismatch-max-score", type=float, default=0.50)
    parser.add_argument("--ok-min-score", type=float, default=0.58)
    parser.add_argument("--progress", type=int, default=0, help="Print progress every N groups to stderr.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.root.exists():
        print(f"OpenPose root does not exist: {args.root}", file=sys.stderr)
        return 2
    summary, rows = audit(args)
    write_json_report(args.report, summary, rows)
    write_csv_report(args.csv_report, rows)
    write_html_report(args.html_report, args.root.resolve(), summary, rows)

    print(f"Audited {summary['groups']} asset groups")
    print(f"Reported rows: {summary['reported_rows']}")
    for key in ("mirror_candidate", "mismatch_candidate", "ambiguous", "not_scored"):
        print(f"{key}: {summary.get(key, 0)}")
    print(f"JSON report: {args.report}")
    print(f"CSV report: {args.csv_report}")
    print(f"HTML report: {args.html_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
