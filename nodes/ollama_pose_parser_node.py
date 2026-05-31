import json
from typing import Any, Dict, List

try:
    from ..core.openpose_io import extract_first_json, normalize_attributes, normalize_token
except ImportError:
    from core.openpose_io import extract_first_json, normalize_attributes, normalize_token


ATTRIBUTE_ALIASES = {
    "arms_up": "hand_up",
    "hands_up": "hand_up",
    "arm_up": "hand_up",
    "legs_spread": "legs_open",
    "spread_legs": "legs_open",
    "open_legs": "legs_open",
    "crossed_legs": "legs_crossed",
    "leg_crossed": "legs_crossed",
    "hands_on_floor": "hands_on_floor",
    "hand_on_face": "hand_near_face",
    "thinking_pose": "thinking",
}

POSE_ALIASES = {
    "stand": "standing",
    "standing_up": "standing",
    "sit": "sitting",
    "seated": "sitting",
    "kneel": "kneeling",
    "on_knees": "kneeling",
    "laying": "lying",
    "lay": "lying",
    "all_fours": "kneeling",
}


class OllamaPoseParserNode:
    """Validate and normalize Ollama pose-intent JSON."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ollama_output": ("STRING", {"multiline": True, "default": ""}),
                "default_variant": ("STRING", {"default": "base", "multiline": False}),
                "max_people": ("INT", {"default": 4, "min": 1, "max": 10}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("structure_json", "parser_report_json")
    FUNCTION = "parse"
    CATEGORY = "OPM/AI"

    @staticmethod
    def _normalize_attributes(values: Any) -> List[str]:
        normalized = []
        seen = set()
        for attribute in normalize_attributes(values):
            mapped = ATTRIBUTE_ALIASES.get(attribute, attribute)
            if mapped and mapped not in seen:
                normalized.append(mapped)
                seen.add(mapped)
        return normalized

    @staticmethod
    def _normalize_person(raw: Dict[str, Any], default_variant: str) -> Dict[str, Any]:
        pose = normalize_token(raw.get("pose") or raw.get("body_position"))
        pose = POSE_ALIASES.get(pose, pose)

        subpose = normalize_token(raw.get("subpose") or raw.get("pose_detail") or raw.get("detail"))
        variant = normalize_token(raw.get("variant") or default_variant)
        gender = normalize_token(raw.get("gender"))
        attributes = OllamaPoseParserNode._normalize_attributes(
            raw.get("attributes") or raw.get("pose_attributes") or raw.get("actions")
        )

        result = {
            "role": str(raw.get("role") or raw.get("name") or "").strip(),
            "pose": pose,
            "variant": variant,
            "subpose": subpose,
            "gender": gender,
            "attributes": attributes,
        }

        for key in ("position", "interaction", "negative_attributes", "id", "pose_id"):
            if key in raw:
                result[key] = raw[key]

        return {key: value for key, value in result.items() if value not in ("", [], None)}

    @staticmethod
    def _extract_people(data: Any) -> List[Dict[str, Any]]:
        if isinstance(data, dict):
            for key in ("people", "persons", "characters", "subjects"):
                if isinstance(data.get(key), list):
                    return [item for item in data[key] if isinstance(item, dict)]
            return [data]
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
        return []

    def parse(self, ollama_output, default_variant, max_people):
        try:
            json_text = extract_first_json(ollama_output)
            data = json.loads(json_text)
        except Exception as exc:
            report = {"error": f"Could not parse Ollama JSON: {exc}"}
            return ("", json.dumps(report, indent=2, ensure_ascii=False))

        raw_people = self._extract_people(data)[:max_people]
        people = [self._normalize_person(person, default_variant) for person in raw_people]
        people = [person for person in people if person.get("pose") or person.get("attributes") or person.get("id") or person.get("pose_id")]

        payload = {
            "schema": "pal_pose_intent/v1",
            "people": people,
        }
        if isinstance(data, dict):
            if data.get("scene"):
                payload["scene"] = data.get("scene")
            if data.get("layout"):
                payload["layout"] = data.get("layout")

        report = {
            "schema": "pal_ollama_parser_report/v1",
            "input_people": len(raw_people),
            "output_people": len(people),
            "default_variant": normalize_token(default_variant),
        }
        return (json.dumps(payload, ensure_ascii=False), json.dumps(report, indent=2, ensure_ascii=False))
