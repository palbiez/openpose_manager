# OPM Test Workflows

Import these JSON files through ComfyUI's workflow loader.

- `opm_t2i_pose_by_id_pony.json`: Pony T2I workflow using `OPM Pose By ID` and `OPM OpenPose Renderer` as OpenPose ControlNet input.
- `opm_t2i_ollama_pose_selection_pony.json`: Pony T2I workflow based on the compact `pony_t2i.json` graph, with Ollama generating pose intent JSON for OPM pose selection.
- `opm_component_pose_by_id_preview.json`: Smoke test for `OPM Pose By ID` and `OPM OpenPose Renderer`.
- `opm_component_selector_preview.json`: Smoke test for `OPM Pose Selector`.
- `opm_component_parser_structure_preview.json`: Smoke test for `OPM Ollama Pose Parser`, `OPM Pose From Structure`, and rendering.
- `opm_component_matcher_preview.json`: Smoke test for `OPM Pose Matcher` and rendering.

The reusable Ollama system prompt is in:

```text
system_prompts/opm_ollama_pose_selection_system_prompt.txt
```

The T2I workflows expect the local model names already used in your setup:

```text
pony\ponyRealism_V23ULTRA.safetensors
openposeSDXL_v10.safetensors
```
