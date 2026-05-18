#!/usr/bin/env python3
"""List OpenPose assets that do not have a matching *_depth.png file."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional


DEFAULT_OPENPOSE_ROOT = Path(
    r"C:\EasyDiffusion\stable-diffusion\stable-diffusion-webui\models\openpose"
)

POSE_FILE_SUFFIX_RE = re.compile(
    r"_(dup\d+|duplicate\d*|copy\d*|bone_structure_full|bone_structure|openposefull|openposehand|openpose|"
    r"depthhand|normalhand|cannyhand|line_art|lineart|linart|canny|depth|normal)"
    r"(_[a-z0-9]+)?$",
    re.IGNORECASE,
)


@dataclass
class MissingDepth:
    json_path: str
    expected_depth_path: str
    base_name: str
    folder: str


def strip_pose_file_suffix(stem: str) -> str:
    base = stem.strip()
    while True:
        match = POSE_FILE_SUFFIX_RE.search(base)
        if not match:
            return base.strip()
        base = base[: match.start()].rstrip()


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


def find_missing_depths(root: Path, include_plain_json: bool, relative: bool) -> List[MissingDepth]:
    missing: List[MissingDepth] = []

    for json_path in iter_pose_jsons(root, include_plain_json):
        base_name = strip_pose_file_suffix(json_path.stem)
        if not base_name:
            continue
        if has_matching_depth(json_path.parent, base_name):
            continue

        expected = json_path.parent / f"{base_name}_depth.png"
        if relative:
            json_value = str(json_path.relative_to(root))
            expected_value = str(expected.relative_to(root))
            folder_value = str(json_path.parent.relative_to(root))
        else:
            json_value = str(json_path)
            expected_value = str(expected)
            folder_value = str(json_path.parent)

        missing.append(
            MissingDepth(
                json_path=json_value,
                expected_depth_path=expected_value,
                base_name=base_name,
                folder=folder_value,
            )
        )

    return missing


def write_text(rows: List[MissingDepth], output: Optional[Path]) -> None:
    lines = [row.expected_depth_path for row in rows]
    text = "\n".join(lines)
    if text:
        text += "\n"
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8", newline="\n")
    else:
        print(text, end="")


def write_json(rows: List[MissingDepth], output: Optional[Path]) -> None:
    payload = [asdict(row) for row in rows]
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8", newline="\n")
    else:
        print(text)


def write_csv(rows: List[MissingDepth], output: Optional[Path]) -> None:
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        handle = output.open("w", encoding="utf-8", newline="")
        close_handle = True
    else:
        handle = sys.stdout
        close_handle = False

    try:
        writer = csv.DictWriter(handle, fieldnames=["json_path", "expected_depth_path", "base_name", "folder"])
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    finally:
        if close_handle:
            handle.close()


def parse_args() -> argparse.Namespace:
    env_root = os.environ.get("OPENPOSE_MODELS_PATH")
    default_root = Path(env_root) if env_root else DEFAULT_OPENPOSE_ROOT

    parser = argparse.ArgumentParser(description="List OpenPose JSON files missing matching *_depth.png files.")
    parser.add_argument("--root", type=Path, default=default_root, help="OpenPose dataset root.")
    parser.add_argument(
        "--include-plain-json",
        action="store_true",
        help="Scan all JSON files instead of only *_openpose.json files.",
    )
    parser.add_argument(
        "--format",
        choices=["text", "json", "csv"],
        default="text",
        help="Output format. Text prints one expected depth path per line.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional output file.")
    parser.add_argument("--absolute", action="store_true", help="Print absolute paths instead of paths relative to --root.")
    parser.add_argument(
        "--fail-on-missing",
        action="store_true",
        help="Return exit code 1 when missing depth files are found. Useful for CI checks.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    if not root.exists():
        print(f"OpenPose root does not exist: {root}", file=sys.stderr)
        return 2

    rows = find_missing_depths(root, include_plain_json=args.include_plain_json, relative=not args.absolute)

    if args.format == "json":
        write_json(rows, args.output)
    elif args.format == "csv":
        write_csv(rows, args.output)
    else:
        write_text(rows, args.output)

    print(f"Missing depth files: {len(rows)}", file=sys.stderr)
    return 1 if rows and args.fail_on_missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
