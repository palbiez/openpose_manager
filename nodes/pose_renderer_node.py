import json

try:
    from ..core.openpose_io import (
        draw_people,
        extract_people,
        fit_people_to_canvas,
        image_to_tensor,
        make_pose_payload,
    )
except ImportError:
    from core.openpose_io import (
        draw_people,
        extract_people,
        fit_people_to_canvas,
        image_to_tensor,
        make_pose_payload,
    )


class PoseOpenPoseRendererNode:
    """Render OPM/OpenPose keypoint JSON into a ComfyUI IMAGE tensor."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_json": ("STRING", {"multiline": True, "default": ""}),
                "width": ("INT", {"default": 768, "min": 64, "max": 4096, "step": 8}),
                "height": ("INT", {"default": 768, "min": 64, "max": 4096, "step": 8}),
                "layout": (["fit_each_person", "preserve_coordinates"], {"default": "fit_each_person"}),
                "style": (["openpose_color", "white"], {"default": "openpose_color"}),
                "line_width": ("INT", {"default": 4, "min": 1, "max": 24}),
                "point_radius": ("INT", {"default": 4, "min": 1, "max": 24}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "rendered_pose_json")
    FUNCTION = "render"
    CATEGORY = "OPM/Render"

    def render(self, pose_json, width, height, layout, style, line_width, point_radius):
        try:
            people = extract_people(pose_json)
        except Exception as exc:
            print(f"[PoseRenderer] Invalid pose JSON: {exc}")
            people = []

        if layout == "fit_each_person":
            people = fit_people_to_canvas(people, width, height)

        canvas = draw_people(
            people,
            width,
            height,
            line_width=line_width,
            point_radius=point_radius,
            style=style,
        )
        rendered_payload = make_pose_payload(
            [
                {
                    **person.metadata,
                    "keypoints": person.keypoints,
                }
                for person in people
            ]
        )
        return (image_to_tensor(canvas), json.dumps(rendered_payload, ensure_ascii=False))
