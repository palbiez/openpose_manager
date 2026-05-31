#!/usr/bin/env python3
"""Run lightweight import and node-contract checks without pytest."""

import json
import sys
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parent.parent
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))

from core.openpose_io import extract_people
from nodes.ollama_pose_parser_node import OllamaPoseParserNode
from nodes.pose_browser_node import PoseBrowserLauncherNode, PoseLoadByIdNode
from nodes.pose_from_structure_node import PoseFromStructureNode
from nodes.pose_matcher_node import PoseMatcherNode
from nodes.pose_renderer_node import PoseOpenPoseRendererNode
from nodes.pose_selector_node import PoseSelectorNode


def main() -> int:
    parser = OllamaPoseParserNode()
    structure_json, _ = parser.parse(
        '{"people":[{"role":"subject","pose":"seated","attributes":["crossed_legs"]}]}',
        "base",
        4,
    )
    structure = json.loads(structure_json)
    assert structure["people"][0]["pose"] == "sitting"
    assert structure["people"][0]["attributes"] == ["legs_crossed"]

    keypoints = []
    for index in range(18):
        keypoints.extend([100 + index * 2, 100 + index * 3, 1.0])

    payload = {
        "schema": "pal_pose_selection/v1",
        "people": [{"id": 1, "pose": "standing", "keypoints": keypoints}],
    }
    assert len(extract_people(payload)) == 1

    renderer = PoseOpenPoseRendererNode()
    image, _ = renderer.render(json.dumps(payload), 256, 256, "fit_each_person", "white", 2, 2)
    assert tuple(image.shape) == (1, 256, 256, 3)

    categories = {
        OllamaPoseParserNode.CATEGORY,
        PoseFromStructureNode.CATEGORY,
        PoseSelectorNode.CATEGORY,
        PoseLoadByIdNode.CATEGORY,
        PoseOpenPoseRendererNode.CATEGORY,
        PoseMatcherNode.CATEGORY,
        PoseBrowserLauncherNode.CATEGORY,
    }
    assert categories == {
        "OPM/AI",
        "OPM/Selection",
        "OPM/Browser",
        "OPM/Render",
        "OPM/Analysis",
    }

    print("Smoke checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
