# Node Reference

## OPM Ollama Pose Parser

Class: `OllamaPoseParserNode`

Sidebar: `OPM/AI`

Normalizes JSON returned by Ollama into `pal_pose_intent/v1`.

Inputs:

- `ollama_output`: raw Ollama text or JSON.
- `default_variant`: variant to apply when a person has no variant.
- `max_people`: maximum number of person specs to keep.

Outputs:

- `structure_json`: normalized pose intent JSON.
- `parser_report_json`: parser status and counts.

## OPM Pose From Structure

Class: `PoseFromStructureNode`

Sidebar: `OPM/Selection`

Selects real database poses from normalized person specs.

Inputs:

- `structure_json`: object with `people[]`, usually from the Ollama parser.
- `num_people`: maximum people to match.
- `match_strictness`: `balanced`, `strict`, or `loose`.
- `seed_control`: `randomize`, `fixed`, or `incremental`.
- `seed`: deterministic seed when seed control is fixed.

Outputs:

- `pose_json`: `pal_pose_selection/v1` with selected people and keypoints.
- `match_report_json`: selected IDs, candidate counts, scores, and request echo.

## OPM Pose Selector

Class: `PoseSelectorNode`

Sidebar: `OPM/Selection`

Manual pose selection by ID or category filters.

Inputs:

- `selection_mode`: `filters` or `pose_id`.
- `pose_id`: direct registry ID when using ID mode.
- `pose`, `gender`, `variant`, `subpose`: registry filter values.
- `attribute_query`: comma-separated attributes.
- `attribute_mode`: `prefer`, `require_all`, or `ignore`.
- `seed_control`, `seed`: deterministic selection controls.

Outputs:

- `pose_json`: selected pose as `pal_pose_selection/v1`.
- `metadata_json`: selection report.

## OPM Pose By ID

Class: `PoseLoadByIdNode`

Sidebar: `OPM/Browser`

Loads a pose by browser/registry ID.

Inputs:

- `pose_id`: registry ID.
- `preferred_image`: `auto`, `depth`, `bone_structure`, or `bone_structure_full`.

Outputs:

- `pose_json`: selected pose with keypoints.
- `selected_image_path`: preferred image path.
- `depth_image_path`: depth preview path when available.
- `bone_image_path`: bone structure path when available.
- `metadata_json`: pose metadata.

## OPM OpenPose Renderer

Class: `PoseOpenPoseRendererNode`

Sidebar: `OPM/Render`

Renders OpenPose/COCO/OPM keypoint JSON into a ComfyUI `IMAGE`.

Inputs:

- `pose_json`: OPM selection JSON, OpenPose JSON, plain keypoint list, or matcher result list.
- `width`, `height`: output size.
- `layout`: `fit_each_person` or `preserve_coordinates`.
- `style`: `openpose_color` or `white`.
- `line_width`, `point_radius`: render thickness.

Outputs:

- `image`: ComfyUI image tensor.
- `rendered_pose_json`: final keypoint payload after optional fitting.

## OPM Pose Matcher

Class: `PoseMatcherNode`

Sidebar: `OPM/Analysis`

Compares incoming keypoints to the reference database.

Inputs:

- `coco_keypoints`: JSON keypoints or OpenPose-like payload.
- `top_k`: number of matches.

Outputs:

- `matches_json`: list of matched poses with IDs, scores, metadata, and keypoints.

## OPM OpenPose Browser Launcher

Class: `PoseBrowserLauncherNode`

Sidebar: `OPM/Browser`

Starts the local browser server manually.

Output:

- `status`: launch status text.

## Removed Legacy Nodes

The old duplicate aliases and path-only legacy nodes are no longer registered in ComfyUI. Use `OPM Pose By ID` and `OPM OpenPose Renderer` for current workflows.
