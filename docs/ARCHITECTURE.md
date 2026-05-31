# Architecture

## Goal

OpenPose Manager (OPM) produces deterministic OpenPose skeleton inputs for ComfyUI workflows. It is designed for Flux / Flux2 pipelines where text prompts alone are not reliable enough for multi-person anatomy and spatial relationships.

The project uses real OpenPose data as the geometry source of truth. Ollama or another LLM should describe pose intent, not generate raw keypoints.

## Target Pipeline

```text
User prompt
-> Ollama scene/person extraction
-> OPM Ollama Pose Parser
-> OPM Pose From Structure
-> OPM OpenPose Renderer
-> ControlNet / Flux workflow
```

Manual workflows can skip Ollama:

```text
Pose Browser ID
-> OPM Pose By ID
-> OPM OpenPose Renderer
-> ControlNet / Flux workflow
```

## Components

### core

`core/pose_registry.py`

Scans `models/openpose`, groups related files, assigns stable IDs, reads keypoints lazily from JSON or PNG metadata, and builds `pose_registry_cache.json`.

`core/pose_similarity_matcher.py`

Normalizes incoming keypoints and compares them to the real pose database. OpenPose18 and COCO17 inputs are converted to a shared canonical 17-point representation before distance comparison.

`core/openpose_io.py`

Contains shared JSON parsing, keypoint layout detection, OpenPose18/COCO17 conversion, multi-person fitting, and skeleton rendering helpers.

`core/pose_attributes.py`

Assigns attributes such as `legs_open`, `hand_up`, `thinking`, `arms_crossed`, and `legs_crossed` from keypoint geometry and optional filename hints.

### nodes

ComfyUI nodes are thin wrappers around `core/`. They should not contain dataset scanning logic or image parsing beyond node-specific input/output handling.

### scripts

CLI scripts are maintenance tools for rebuilding the cache, enriching attributes, and running smoke checks.

### web

The pose browser is a local HTML UI backed by the registry API. Inside ComfyUI it is exposed at `/poses`; the standalone FastAPI server on port `8189` remains available as a fallback. The UI shows depth previews by default and switches to bone structure previews on hover.

## Data Contract

Nodes that pass selected poses should use:

```json
{
  "schema": "pal_pose_selection/v1",
  "seed": 123,
  "people": [
    {
      "id": 1,
      "pose": "sitting",
      "variant": "base",
      "subpose": "floor",
      "attributes": ["legs_crossed"],
      "keypoints": [0, 0, 0]
    }
  ]
}
```

The renderer also accepts plain OpenPose JSON with `people[].pose_keypoints_2d` and older match-result lists with `keypoints`.

## Cache Policy

The registry cache stores metadata and source file paths only. It does not persist keypoints. Keypoints are copied from the original JSON when needed, which keeps the cache smaller and avoids duplicating pose data.

## Design Rules

- Keep real pose files as the authoritative data.
- Keep core logic independent from ComfyUI where possible.
- Keep nodes focused on stable input/output contracts.
- Prefer attribute-weighted matching over synthetic pose generation.
- Preserve compatibility wrappers while the project is still evolving.
