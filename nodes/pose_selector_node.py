import json
import random
from typing import Dict, List

try:
    from ..core.openpose_io import make_pose_payload, normalize_attributes, normalize_token
    from ..core.pose_registry import get_registry
except ImportError:
    from core.openpose_io import make_pose_payload, normalize_attributes, normalize_token
    from core.pose_registry import get_registry


_cached_options = None


def get_pose_options():
    global _cached_options
    if _cached_options is not None:
        return _cached_options

    try:
        registry = get_registry()
        poses = {item.get("pose", "unknown") for item in registry.poses}
        genders = {item.get("gender", "unknown") for item in registry.poses}
        variants = {item.get("variant", "base") for item in registry.poses}
        subposes = {item.get("subpose", "default") for item in registry.poses}
        attributes = {
            attribute
            for item in registry.poses
            for attribute in item.get("attributes", [])
            if attribute
        }
        _cached_options = {
            "poses": sorted(poses) or ["standing"],
            "genders": ["any"] + sorted(genders),
            "variants": sorted(variants) or ["base"],
            "subposes": sorted(subposes) or ["default"],
            "attributes": sorted(attributes),
        }
    except Exception:
        _cached_options = {
            "poses": ["standing", "sitting", "kneeling", "lying"],
            "genders": ["any", "f", "m", "unknown"],
            "variants": ["base", "nsfw"],
            "subposes": ["neutral", "default"],
            "attributes": [],
        }

    return _cached_options


def _next_seed(node, seed_control, seed):
    if seed_control == "fixed":
        final_seed = int(seed)
    elif seed_control == "incremental":
        final_seed = node.last_seed + 1
    else:
        final_seed = random.randint(0, 0xFFFFFFFFFFFFFFFF)
    node.last_seed = final_seed
    return final_seed


def _pose_to_person(registry, item: Dict) -> Dict:
    return {
        "id": item.get("id"),
        "pose": item.get("pose"),
        "gender": item.get("gender"),
        "variant": item.get("variant"),
        "subpose": item.get("subpose"),
        "attributes": item.get("attributes", []),
        "source_file": item.get("source_file"),
        "json_path": item.get("json_path"),
        "score": 0.0,
        "keypoints": registry.get_keypoints_by_id(item["id"]) or [],
    }


class PoseSelectorNode:
    """Select a real pose by ID, category filters, and optional attributes."""

    @classmethod
    def INPUT_TYPES(cls):
        options = get_pose_options()
        return {
            "required": {
                "selection_mode": (["filters", "pose_id"], {"default": "filters"}),
                "pose_id": ("INT", {"default": 0, "min": 0}),
                "pose": (options["poses"], {"default": options["poses"][0]}),
                "gender": (options["genders"], {"default": options["genders"][0]}),
                "variant": (options["variants"], {"default": options["variants"][0]}),
                "subpose": (options["subposes"], {"default": options["subposes"][0]}),
                "attribute_query": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "placeholder": "Comma-separated attributes, e.g. legs_open, hand_up",
                    },
                ),
                "attribute_mode": (["prefer", "require_all", "ignore"], {"default": "prefer"}),
                "seed_control": (["randomize", "fixed", "incremental"], {"default": "randomize"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("pose_json", "metadata_json")
    FUNCTION = "select"
    CATEGORY = "OPM/Selection"

    def __init__(self):
        self.registry = get_registry()
        self.last_seed = 0

    def _score_candidate(self, item, attributes: List[str], attribute_mode: str) -> int:
        item_attributes = set(normalize_attributes(item.get("attributes", [])))
        if attribute_mode == "require_all" and not set(attributes).issubset(item_attributes):
            return -1

        score = 0
        if attributes and attribute_mode != "ignore":
            score += len(set(attributes) & item_attributes) * 10
        return score

    def _select_by_filters(self, pose, gender, variant, subpose, attributes, attribute_mode):
        pose = normalize_token(pose)
        gender = normalize_token(gender)
        variant = normalize_token(variant)
        subpose = normalize_token(subpose)

        candidates = []
        for item in self.registry.poses:
            if normalize_token(item.get("pose")) != pose:
                continue
            if gender and gender != "any" and normalize_token(item.get("gender")) != gender:
                continue
            if variant and normalize_token(item.get("variant")) != variant:
                continue
            if subpose and normalize_token(item.get("subpose")) != subpose:
                continue

            score = self._score_candidate(item, attributes, attribute_mode)
            if score >= 0:
                candidates.append((score, item))

        if not candidates and attributes and attribute_mode == "prefer":
            return self._select_by_filters(pose, gender, variant, subpose, [], "ignore")

        if not candidates:
            return None, []

        best_score = max(score for score, _ in candidates)
        best = [item for score, item in candidates if score == best_score]
        return random.choice(best), best

    def select(self, selection_mode, pose_id, pose, gender, variant, subpose, attribute_query, attribute_mode, seed_control, seed):
        final_seed = _next_seed(self, seed_control, seed)
        random.seed(final_seed)
        attributes = normalize_attributes(attribute_query)

        selected = None
        candidates = []
        if selection_mode == "pose_id" and int(pose_id) > 0:
            selected = self.registry.get_pose_by_id(int(pose_id))
            candidates = [selected] if selected else []
        else:
            selected, candidates = self._select_by_filters(pose, gender, variant, subpose, attributes, attribute_mode)

        if not selected:
            report = {
                "error": "No matching pose found",
                "selection_mode": selection_mode,
                "pose_id": pose_id,
                "pose": pose,
                "gender": gender,
                "variant": variant,
                "subpose": subpose,
                "attributes": attributes,
            }
            return ("", json.dumps(report, indent=2, ensure_ascii=False))

        person = _pose_to_person(self.registry, selected)
        payload = make_pose_payload([person], seed=final_seed)
        report = {
            "selected_id": selected.get("id"),
            "candidate_count": len(candidates),
            "seed": final_seed,
            "selection_mode": selection_mode,
            "requested_attributes": attributes,
            "selected_attributes": selected.get("attributes", []),
        }

        return (json.dumps(payload, ensure_ascii=False), json.dumps(report, indent=2, ensure_ascii=False))
