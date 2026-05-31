# OpenPose Manager for ComfyUI

OpenPose Manager (OPM) is a ComfyUI custom node package for selecting, matching, and rendering real OpenPose skeletons from a local pose library.

Repository:

```text
https://github.com/palbiez/openpose_manager
```

The intended pipeline is:

```text
User prompt
-> Ollama structure extraction
-> normalized pose intent
-> real pose selection from the OpenPose database
-> OpenPose skeleton render
-> ControlNet / Flux image generation
```

The project avoids freeform keypoint generation. It uses real pose data as the geometric source of truth so multi-person scenes stay more stable.

## Features

- Browse and select local OpenPose assets from ComfyUI.
- Match structured pose intent to real database poses.
- Render selected OpenPose keypoints for downstream ControlNet / Flux workflows.
- Import downloaded pose collections into the OpenPose Manager folder layout.
- Audit depth, bone-structure, and OpenPose JSON consistency with JSON, CSV, and HTML reports.
- Audit image alignment between OpenPose JSON / bone previews and rendered depth, normal, or lineart companions.
- Reconstruct missing `*_openpose.json` files from color-coded `*_bone_structure.png` files when enough keypoints can be recovered.
- Assign geometry-based pose attributes for filtering and matching.

## Project Layout

```text
core/                 Shared registry, matching, OpenPose parsing, rendering helpers
nodes/                ComfyUI node implementations
scripts/              CLI maintenance tools
web/pose_browser/     Local pose browser UI
docs/                 Architecture, schemas, node reference
tests/smoke/          Lightweight contract tests
```

Root-level modules such as `pose_registry.py` and `build_pose_cache.py` are compatibility wrappers for older imports and commands.

## Installation

1. Place this folder in `ComfyUI/custom_nodes/`.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Put pose files under:

```text
ComfyUI/models/openpose/
```

The scanner supports nested OpenPose folders such as:

```text
openpose/standing/M/base/bed_mirror_selfie/bed_mirror_selfie_000_bone_structure.png
openpose/standing/M/base/bed_mirror_selfie/bed_mirror_selfie_000_depth.png
openpose/standing/M/base/bed_mirror_selfie/bed_mirror_selfie_000_openpose.json
```

Preview images prefer `*_depth.png`. If no depth image exists, `*_bone_structure.png` is used.

## Cache

Build or refresh the registry cache:

```bash
python scripts/build_pose_cache.py
```

The legacy command still works:

```bash
python build_pose_cache.py
```

Clean the cache:

```bash
python scripts/build_pose_cache.py --clean
```

## Pose Collection Import

Import downloaded pose collections into the OpenPose Manager layout:

```powershell
python scripts/import_pose_collections.py --source "<downloaded_pose_folder>" --output-root "<ComfyUI>/models/openpose"
```

The source folder can also be set through `OPENPOSE_IMPORT_SOURCE`:

```powershell
$env:OPENPOSE_IMPORT_SOURCE="<downloaded_pose_folder>"
python scripts/import_pose_collections.py --output-root "<ComfyUI>/models/openpose"
```

Use `--render-bone` when the source images are not already OpenPose-style skeleton previews.

## Pose Attributes

Assign automatic pose attributes from keypoint geometry:

```powershell
python scripts/auto_pose_attributes.py --root "$env:USERPROFILE\Documents\ComfyUI\models\openpose" --write
```

macOS / Linux example:

```bash
python scripts/auto_pose_attributes.py --root "$HOME/ComfyUI/models/openpose" --write
```

Attributes are written into OpenPose JSON metadata as `meta.auto_attributes` and `meta.attributes`.

## Pose Asset Audit

Check depth, bone-structure, and OpenPose JSON consistency:

```powershell
python scripts/audit_pose_assets.py --only-issues
```

Limit the audit to one folder:

```powershell
python scripts/audit_pose_assets.py --scope sitting/F/nsfw/sitting --only-issues
```

The script writes local JSON, CSV, and HTML reports in the plugin folder. To create missing `*_openpose.json` files from color-coded `*_bone_structure.png` files:

```powershell
python scripts/audit_pose_assets.py --write-missing-json --min-bone-points 18
```

## Pose Image Alignment Audit

Check whether rendered companion images are horizontally mirrored or mismatched relative to `*_bone_structure.png` and `*_openpose.json`:

```powershell
$env:OPENPOSE_MODELS_PATH="<ComfyUI>\models\openpose"
python scripts/audit_pose_image_alignment.py --only-issues
```

The script writes local JSON, CSV, and HTML reports in the plugin folder:

```text
pose_image_alignment_audit.json
pose_image_alignment_audit.csv
pose_image_alignment_audit.html
```

These reports are local maintenance output and are ignored by Git. The HTML report is the easiest way to inspect candidates visually. The relevant classifications are:

- `mirror_candidate`: the horizontally flipped JSON/bone pose overlaps the rendered image better than the original orientation.
- `mismatch_candidate`: neither the original nor the flipped orientation overlaps well, so the files likely do not belong together.
- `ambiguous`: possible issue, but the score is not strong enough for automatic repair.

To repair only `mirror_candidate` rows, run a dry run first:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\repair_mirrored_pose_renders.ps1
```

Apply the fix with backups:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\repair_mirrored_pose_renders.ps1 -Apply -Backup
```

Ambiguous candidates are not repaired by default. Repair a known ambiguous pose by name pattern:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\repair_mirrored_pose_renders.ps1 -Classification ambiguous -BaseNamePattern "*dance_03" -Apply -Backup
```

Or dry-run ambiguous candidates above explicit score thresholds:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\repair_mirrored_pose_renders.ps1 -Classification ambiguous -MinFlipDelta 0.14 -MinFlippedScore 0.66
```

The repair script reads `pose_image_alignment_audit.csv`, uses `OPENPOSE_MODELS_PATH` when no `-Root` is passed, and requires ImageMagick. It flips `depth`, `normal`, and `lineart` files by default. It does not modify `*_bone_structure.png` or `*_openpose.json`.

## Pose Browser

When loaded inside ComfyUI, the browser is registered under the ComfyUI server:

```text
http://127.0.0.1:8188/poses
```

If the ComfyUI route integration is not available, the standalone browser server can still run on:

```text
http://127.0.0.1:8189
```

Environment variables:

Windows PowerShell:

```powershell
$env:OPENPOSE_MODELS_PATH="$env:USERPROFILE\Documents\ComfyUI\models\openpose"
$env:OPENPOSE_BROWSER_HOST="0.0.0.0"
$env:OPENPOSE_BROWSER_PORT="8189"
$env:OPENPOSE_BROWSER_AUTOSTART="1"
```

Windows Command Prompt:

```bat
set OPENPOSE_MODELS_PATH=%USERPROFILE%\Documents\ComfyUI\models\openpose
set OPENPOSE_BROWSER_HOST=0.0.0.0
set OPENPOSE_BROWSER_PORT=8189
set OPENPOSE_BROWSER_AUTOSTART=1
```

macOS / Linux:

```bash
export OPENPOSE_MODELS_PATH="$HOME/ComfyUI/models/openpose"
export OPENPOSE_BROWSER_HOST="0.0.0.0"
export OPENPOSE_BROWSER_PORT="8189"
export OPENPOSE_BROWSER_AUTOSTART="1"
```

Optional PNG metadata fallback:

```bash
export EXIFTOOL_PATH="/usr/local/bin/exiftool"
```

## Main Nodes

- `OPM Ollama Pose Parser`: validates and normalizes Ollama JSON output.
- `OPM Pose From Structure`: selects real database poses from normalized structure JSON.
- `OPM Pose Selector`: manually selects a pose by ID, filters, and attributes.
- `OPM Pose By ID`: loads pose JSON, image paths, and metadata from a browser pose ID.
- `OPM OpenPose Renderer`: renders OPM/OpenPose keypoint JSON to a ComfyUI `IMAGE`.
- `OPM Pose Matcher`: finds similar database poses for incoming keypoints.
- `OPM OpenPose Browser Launcher`: manually starts the browser server.

## ComfyUI Sidebar

Nodes are grouped under the `OPM` sidebar folder:

- `OPM/AI`: prompt or LLM structure parsing.
- `OPM/Selection`: pose lookup and matching from structured intent.
- `OPM/Browser`: pose browser ID loading and browser launch.
- `OPM/Render`: OpenPose skeleton rendering.
- `OPM/Analysis`: similarity matching utilities.

See [docs/NODE_REFERENCE.md](docs/NODE_REFERENCE.md) for input and output contracts.

## Development Checks

Run lightweight smoke checks:

```bash
python scripts/smoke_check.py
```

Or with pytest:

```bash
python -m pytest tests/smoke
```

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Ollama Schema](docs/OLLAMA_SCHEMA.md)
- [Node Reference](docs/NODE_REFERENCE.md)
- [Dataset Gaps](docs/DATASET_GAPS.md)
