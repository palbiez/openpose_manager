import json
import random
from typing import Any, Dict, List, Tuple

try:
    from ..core.openpose_io import make_pose_payload, normalize_attributes, normalize_token
    from ..core.pose_registry import get_registry
    from .pose_selector_node import _next_seed, _pose_to_person
except ImportError:
    from core.openpose_io import make_pose_payload, normalize_attributes, normalize_token
    from core.pose_registry import get_registry
    from nodes.pose_selector_node import _next_seed, _pose_to_person


def _parse_structure(structure_json: str) -> List[Dict[str, Any]]:
    data = json.loads(structure_json)
    if isinstance(data, dict) and isinstance(data.get("people"), list):
        return data["people"]
    if isinstance(data, list):
        return data
    raise ValueError("Expected a JSON object with people[] or a list of person specs.")


class PoseFromStructureNode:
    """Convert normalized scene/person structure into real DB poses."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "structure_json": ("STRING", {"multiline": True, "default": ""}),
                "num_people": ("INT", {"default": 1, "min": 1, "max": 10}),
                "match_strictness": (["balanced", "strict", "loose"], {"default": "balanced"}),
                "seed_control": (["randomize", "fixed", "incremental"], {"default": "randomize"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("pose_json", "match_report_json")
    FUNCTION = "convert"
    CATEGORY = "OPM/Selection"

    def __init__(self):
        self.registry = get_registry()
        self.last_seed = 0

    @staticmethod
    def _candidate_score(item: Dict[str, Any], spec: Dict[str, Any], strictness: str) -> int:
        requested_pose = normalize_token(spec.get("pose"))
        requested_variant = normalize_token(spec.get("variant"))
        requested_subpose = normalize_token(spec.get("subpose"))
        requested_gender = normalize_token(spec.get("gender"))
        requested_attributes = normalize_attributes(spec.get("attributes"))
        negative_attributes = set(normalize_attributes(spec.get("negative_attributes")))
        item_attributes = set(normalize_attributes(item.get("attributes", [])))

        item_pose = normalize_token(item.get("pose"))
        item_variant = normalize_token(item.get("variant"))
        item_subpose = normalize_token(item.get("subpose"))
        item_gender = normalize_token(item.get("gender"))

        score = 0

        if requested_pose:
            if item_pose != requested_pose:
                return -1 if strictness != "loose" else 0
            score += 100
        if requested_variant:
            if item_variant == requested_variant:
                score += 30
            elif strictness == "strict":
                return -1
        if requested_subpose:
            if item_subpose == requested_subpose:
                score += 45
            elif strictness == "strict":
                return -1
        if requested_gender and requested_gender not in {"unknown", "any"}:
            if item_gender == requested_gender:
                score += 15
            elif strictness == "strict":
                return -1

        matched_attributes = set(requested_attributes) & item_attributes
        missing_attributes = set(requested_attributes) - item_attributes
        forbidden_attributes = negative_attributes & item_attributes

        score += len(matched_attributes) * 18
        score -= len(missing_attributes) * (14 if strictness == "balanced" else 7)
        score -= len(forbidden_attributes) * 30

        if strictness == "strict" and missing_attributes:
            return -1
        return score

    def _find_match(self, spec: Dict[str, Any], strictness: str) -> Tuple[Dict[str, Any], List[Tuple[int, Dict[str, Any]]]]:
        pose_id = spec.get("id") or spec.get("pose_id")
        if pose_id:
            try:
                item = self.registry.get_pose_by_id(int(pose_id))
            except Exception:
                item = None
            return item, [(1000, item)] if item else []

        scored = []
        for item in self.registry.poses:
            score = self._candidate_score(item, spec, strictness)
            if score >= 0:
                scored.append((score, item))

        if not scored:
            return None, []

        scored.sort(key=lambda pair: pair[0], reverse=True)
        top_score = scored[0][0]
        best_pool = [item for score, item in scored if score == top_score]
        return random.choice(best_pool), scored

    def convert(self, structure_json, num_people, match_strictness, seed_control, seed):
        final_seed = _next_seed(self, seed_control, seed)
        random.seed(final_seed)

        try:
            specs = _parse_structure(structure_json)
        except Exception as exc:
            report = {"error": f"Invalid structure JSON: {exc}"}
            return ("", json.dumps(report, indent=2, ensure_ascii=False))

        selected_people = []
        matches = []
        for index, spec in enumerate(specs[:num_people]):
            if not isinstance(spec, dict):
                matches.append({"index": index, "error": "Person spec is not an object"})
                continue

            selected, scored = self._find_match(spec, match_strictness)
            if not selected:
                matches.append({"index": index, "error": "No matching pose", "request": spec})
                continue

            person = _pose_to_person(self.registry, selected)
            if spec.get("role"):
                person["role"] = spec.get("role")
            if spec.get("position"):
                person["position"] = spec.get("position")
            selected_people.append(person)
            matches.append(
                {
                    "index": index,
                    "selected_id": selected.get("id"),
                    "selected_pose": {
                        "pose": selected.get("pose"),
                        "variant": selected.get("variant"),
                        "subpose": selected.get("subpose"),
                        "attributes": selected.get("attributes", []),
                    },
                    "candidate_count": len(scored),
                    "top_score": scored[0][0] if scored else None,
                    "request": spec,
                }
            )

        payload = make_pose_payload(selected_people, seed=final_seed)
        report = {
            "schema": "pal_pose_match_report/v1",
            "seed": final_seed,
            "strictness": match_strictness,
            "requested_people": min(len(specs), num_people),
            "matched_people": len(selected_people),
            "matches": matches,
        }
        return (json.dumps(payload, ensure_ascii=False), json.dumps(report, indent=2, ensure_ascii=False))
