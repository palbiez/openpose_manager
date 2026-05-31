import json
try:
    from ..core.pose_similarity_matcher import PoseMatcher
except ImportError:
    from core.pose_similarity_matcher import PoseMatcher


class PoseMatcherNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "coco_keypoints": ("STRING", {}),
                "top_k": ("INT", {"default": 2})
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("matches_json",)
    FUNCTION = "run"
    CATEGORY = "OPM/Analysis"

    def __init__(self):
        self.matcher = PoseMatcher()
        print("[PoseMatcherNode] Initialized")

    def run(self, coco_keypoints, top_k):

        try:
            kp = json.loads(coco_keypoints)
        except Exception:
            return ("[]",)

        result = self.matcher.match(kp, top_k=top_k)

        if not result:
            return ("[]",)

        return (json.dumps(result),)
