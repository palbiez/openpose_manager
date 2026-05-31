import json
import os
import subprocess
import sys
from pathlib import Path
try:
    from ..core.openpose_io import make_pose_payload
    from ..core.pose_registry import get_registry
except ImportError:
    from core.openpose_io import make_pose_payload
    from core.pose_registry import get_registry


def _pose_payload(registry, pose_data):
    keypoints = registry.get_keypoints_by_id(pose_data["id"]) or []
    person = {
        "id": pose_data.get("id"),
        "pose": pose_data.get("pose"),
        "gender": pose_data.get("gender"),
        "variant": pose_data.get("variant"),
        "subpose": pose_data.get("subpose"),
        "attributes": pose_data.get("attributes", []),
        "source_file": pose_data.get("source_file"),
        "json_path": pose_data.get("json_path"),
        "keypoints": keypoints,
    }
    return make_pose_payload([person])


def _select_image(pose_data, preferred_image):
    depth_image = pose_data.get("display_image") or pose_data.get("png_path") or ""
    bone_structure = pose_data.get("bone_structure_path") or ""
    bone_structure_full = pose_data.get("bone_structure_full_path") or ""

    if preferred_image == "depth":
        return depth_image
    if preferred_image == "bone_structure":
        return bone_structure or bone_structure_full
    if preferred_image == "bone_structure_full":
        return bone_structure_full or bone_structure
    return depth_image or bone_structure or bone_structure_full


class PoseLoadByIdNode:
    """Resolve a pose ID into keypoints, image paths, and metadata."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_id": ("INT", {"default": 1, "min": 1}),
                "preferred_image": (
                    ["auto", "depth", "bone_structure", "bone_structure_full"],
                    {"default": "auto"},
                ),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("pose_json", "selected_image_path", "depth_image_path", "bone_image_path", "metadata_json")
    FUNCTION = "load"
    CATEGORY = "OPM/Browser"

    def __init__(self):
        self.registry = get_registry()

    def load(self, pose_id, preferred_image):
        pose_data = self.registry.get_pose_by_id(pose_id)
        if not pose_data:
            info = json.dumps({"error": f"Pose ID {pose_id} not found"}, indent=2)
            return ("", "", "", "", info)

        depth_image = pose_data.get("display_image") or pose_data.get("png_path") or ""
        bone_image = pose_data.get("bone_structure_path") or pose_data.get("bone_structure_full_path") or ""
        selected = _select_image(pose_data, preferred_image)
        payload = _pose_payload(self.registry, pose_data)

        metadata = {
            "id": pose_data.get("id"),
            "pose": pose_data.get("pose"),
            "gender": pose_data.get("gender"),
            "variant": pose_data.get("variant"),
            "subpose": pose_data.get("subpose"),
            "attributes": pose_data.get("attributes", []),
            "source_file": pose_data.get("source_file"),
            "json_path": pose_data.get("json_path"),
            "display_image": pose_data.get("display_image"),
            "bone_structure": pose_data.get("bone_structure_path"),
            "bone_structure_full": pose_data.get("bone_structure_full_path"),
        }

        return (
            json.dumps(payload, ensure_ascii=False),
            str(selected),
            str(depth_image),
            str(bone_image),
            json.dumps(metadata, indent=2, ensure_ascii=False),
        )

class PoseBrowserLauncherNode:
    """Launch the OpenPose browser from inside ComfyUI."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {}
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "start_server"
    CATEGORY = "OPM/Browser"

    def start_server(self):
        server_script = Path(__file__).resolve().parent.parent / "pose_browser_server.py"
        if not server_script.exists():
            message = "Pose browser server script not found."
            print(f"[PoseBrowserLauncher] {message}")
            return (message,)

        cmd = [sys.executable, str(server_script)]
        try:
            subprocess.Popen(
                cmd,
                cwd=str(server_script.parent),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
            )
            port = os.getenv("OPENPOSE_BROWSER_PORT", "8189")
            message = f"Standalone OpenPose Browser launched on http://127.0.0.1:{port}"
            print(f"[PoseBrowserLauncher] {message}")
            return (message,)
        except Exception as e:
            message = f"Failed to launch OpenPose Browser: {e}"
            print(f"[PoseBrowserLauncher] {message}")
            return (message,)
