#!/usr/bin/env python3
"""Repair NSFW OpenPose model assets from the original ComfyUI input files.

This is intentionally narrow: it only touches files derived from
NSFW_* folders and writes them into the existing OPM layout:

    <pose>/F/nsfw/<subpose>/<subpose>_NNN_openpose.json
    <pose>/F/nsfw/<subpose>/<subpose>_NNN_bone_structure.png
    <pose>/F/nsfw/<subpose>/<subpose>_NNN_depth.png
    <pose>/F/nsfw/<subpose>/<subpose>_NNN_lineart.png

Duplicate source numbers are resolved by increasing the first digit of the
three-digit number, matching the existing convention (030 -> 130).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


DEFAULT_SOURCE_ROOT = Path(r"C:\Users\firew\Documents\ComfyUI\input\openpose\openpose")
DEFAULT_TARGET_ROOT = Path(r"C:\EasyDiffusion\stable-diffusion\stable-diffusion-webui\models\openpose")


@dataclass(frozen=True)
class SourceAsset:
    source_json: Path
    source_folder: str
    source_stem: str
    pose: str
    subpose: str
    number: str
    target_base: str
    target_dir: Path
    bone: Optional[Path]
    depth: Optional[Path]
    lineart: Optional[Path]


def slugify(value: str) -> str:
    text = value.strip().lower().replace("-", "_").replace(" ", "_")
    text = re.sub(r"[^a-z0-9_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def derive_pose_and_subpose(source_folder: str) -> tuple[str, str]:
    name = slugify(source_folder)
    if name.startswith("nsfw_"):
        name = name[len("nsfw_") :]

    subpose = name
    if "kneel" in name or "all_fours" in name or "squatting" in name:
        pose = "kneeling"
    elif "split" in name or "lying" in name:
        pose = "lying"
    elif "sitting" in name:
        pose = "sitting"
    elif "standing" in name or "suspended" in name:
        pose = "standing"
    else:
        pose = "special"
    return pose, subpose


def source_number(source_stem: str, source_folder: str, subpose: str) -> Optional[str]:
    lowered_stem = source_stem.lower()
    prefixes = [source_folder.lower()]
    if source_folder.lower().startswith("nsfw_"):
        prefixes.append(source_folder.lower()[len("nsfw_") :])
    prefixes.append(subpose.lower())

    suffix: Optional[str] = None
    for prefix in sorted(set(prefixes), key=len, reverse=True):
        if lowered_stem.startswith(prefix):
            suffix = source_stem[len(prefix) :]
            break

    if suffix is None:
        return None
    if suffix == "":
        return "000"

    match = re.fullmatch(r"(?P<number>\d+)(?P<letter>[a-z]?)", suffix, re.IGNORECASE)
    if not match:
        return None
    return match.group("number").zfill(3)


def bump_number(number: str) -> str:
    if not number:
        return "100"
    first = number[0]
    if first.isdigit() and first != "9":
        return str(int(first) + 1) + number[1:]
    return str(int(number) + 100).zfill(len(number))


def variant_rank(path: Path, source_stem: str, kind: str) -> tuple[int, float, str]:
    pattern = re.compile(rf"^{re.escape(source_stem)}(?P<letter>[a-z]?)_{kind}\.png$", re.IGNORECASE)
    match = pattern.match(path.name)
    letter = (match.group("letter") if match else "").lower()
    letter_rank = ord(letter) - ord("a") + 1 if letter else 0
    return letter_rank, path.stat().st_mtime, path.name.lower()


def choose_companion(json_path: Path, kind: str) -> Optional[Path]:
    source_stem = json_path.stem
    candidates = [
        path
        for path in json_path.parent.glob(f"{source_stem}*_{kind}.png")
        if re.match(rf"^{re.escape(source_stem)}[a-z]?_{kind}\.png$", path.name, re.IGNORECASE)
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: variant_rank(path, source_stem, kind))


def choose_bone(json_path: Path) -> Optional[Path]:
    for suffix in ("_bone_structure.png", "_bone_structure_full.png"):
        candidate = json_path.with_name(f"{json_path.stem}{suffix}")
        if candidate.exists():
            return candidate
    return None


def iter_source_assets(source_root: Path, target_root: Path) -> Iterable[SourceAsset]:
    used: set[tuple[Path, str]] = set()
    json_files = sorted(
        source_root.rglob("*.json"),
        key=lambda path: (str(path.parent).lower(), path.name.lower()),
    )

    for json_path in json_files:
        try:
            relative = json_path.relative_to(source_root)
        except ValueError:
            continue
        if len(relative.parts) < 2:
            continue

        source_folder = relative.parts[0]
        if not source_folder.lower().startswith("nsfw_"):
            continue

        pose, subpose = derive_pose_and_subpose(source_folder)
        number = source_number(json_path.stem, source_folder, subpose)
        if number is None:
            continue

        target_dir = target_root / pose / "F" / "nsfw" / subpose
        unique_number = number
        while (target_dir, unique_number) in used:
            unique_number = bump_number(unique_number)
        used.add((target_dir, unique_number))

        target_base = f"{subpose}_{unique_number}"
        yield SourceAsset(
            source_json=json_path,
            source_folder=source_folder,
            source_stem=json_path.stem,
            pose=pose,
            subpose=subpose,
            number=unique_number,
            target_base=target_base,
            target_dir=target_dir,
            bone=choose_bone(json_path),
            depth=choose_companion(json_path, "depth"),
            lineart=choose_companion(json_path, "lineart"),
        )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_if_needed(source: Optional[Path], target: Path, dry_run: bool) -> tuple[str, str]:
    if source is None:
        if target.exists():
            if not dry_run:
                target.unlink()
            return "remove_stale", ""
        return "missing_source", ""
    source_hash = sha256(source)
    if target.exists() and sha256(target) == source_hash:
        return "unchanged", source_hash
    if not dry_run:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    return ("replace" if target.exists() else "create"), source_hash


def build_rows(assets: Iterable[SourceAsset], dry_run: bool) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for asset in assets:
        targets = [
            ("json", asset.source_json, asset.target_dir / f"{asset.target_base}_openpose.json"),
            ("bone_structure", asset.bone, asset.target_dir / f"{asset.target_base}_bone_structure.png"),
            ("depth", asset.depth, asset.target_dir / f"{asset.target_base}_depth.png"),
            ("lineart", asset.lineart, asset.target_dir / f"{asset.target_base}_lineart.png"),
        ]
        for kind, source, target in targets:
            existed_before = target.exists()
            status, source_hash = copy_if_needed(source, target, dry_run=dry_run)
            rows.append(
                {
                    "kind": kind,
                    "status": status,
                    "source": str(source) if source else "",
                    "target": str(target),
                    "target_existed": str(existed_before),
                    "source_hash": source_hash,
                    "source_folder": asset.source_folder,
                    "target_base": asset.target_base,
                }
            )
    return rows


def write_reports(rows: list[dict[str, str]], report_prefix: Path) -> None:
    report_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = report_prefix.with_suffix(".json")
    csv_path = report_prefix.with_suffix(".csv")
    with json_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["kind", "status"])
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict[str, str]]) -> None:
    summary: dict[tuple[str, str], int] = {}
    for row in rows:
        key = (row["kind"], row["status"])
        summary[key] = summary.get(key, 0) + 1
    for (kind, status), count in sorted(summary.items()):
        print(f"{kind:14s} {status:15s} {count}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--target-root", type=Path, default=DEFAULT_TARGET_ROOT)
    parser.add_argument("--report-prefix", type=Path, default=Path("repair_nsfw_openpose_assets_report"))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_root = args.source_root.resolve()
    target_root = args.target_root.resolve()

    if not source_root.exists():
        raise SystemExit(f"Source root does not exist: {source_root}")
    if not target_root.exists():
        raise SystemExit(f"Target root does not exist: {target_root}")

    rows = build_rows(iter_source_assets(source_root, target_root), dry_run=args.dry_run)
    write_reports(rows, args.report_prefix)
    print_summary(rows)
    print(f"Rows: {len(rows)}")
    print(f"Reports: {args.report_prefix.with_suffix('.json')} and {args.report_prefix.with_suffix('.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
