# Ollama Pose Intent Schema

`OPM Ollama Pose Parser` expects Ollama to return JSON. The JSON may be embedded in text; the parser extracts the first JSON object or array it finds.

## Recommended Output

```json
{
  "scene": "two people sitting on the floor",
  "layout": "left and right",
  "people": [
    {
      "role": "left person",
      "pose": "sitting",
      "variant": "base",
      "subpose": "floor",
      "attributes": ["legs_crossed", "hand_near_face"],
      "position": "left"
    },
    {
      "role": "right person",
      "pose": "kneeling",
      "variant": "base",
      "subpose": "one_knee",
      "attributes": ["hand_up"],
      "position": "right"
    }
  ]
}
```

## Fields

`people`

Required. Array of person pose requests.

`role`

Optional. Human-readable role such as `main subject`, `left person`, or `background person`.

`pose`

Recommended. Main category. Common values:

```text
standing, sitting, kneeling, lying, squatting
```

The parser normalizes aliases such as `seated` to `sitting` and `laying` to `lying`.

`variant`

Optional. Dataset variant such as `base` or `nsfw`. If omitted, the node uses `default_variant`.

`subpose`

Optional. Dataset subcategory such as `floor`, `one_knee`, `prone`, `side`, `neutral`.

`attributes`

Optional but important. Attribute hints for weighted DB matching.

Common values:

```text
legs_open
legs_crossed
hand_up
hand_near_face
hand_on_hip
arms_crossed
head_down
torso_lean
thinking
all_fours
hands_on_floor
```

`negative_attributes`

Optional. Attributes to avoid.

`position`

Optional. Preserved as metadata. The current renderer lays people out left to right; advanced spatial placement can build on this field later.

`id` or `pose_id`

Optional. Directly request a known registry pose ID.

## Parser Output

The parser returns:

```json
{
  "schema": "pal_pose_intent/v1",
  "people": [
    {
      "role": "subject",
      "pose": "sitting",
      "variant": "base",
      "attributes": ["legs_crossed"]
    }
  ]
}
```

This output is intended for `OPM Pose From Structure`.

## Prompting Guidance

Ask Ollama for compact JSON only. Example:

```text
Extract pose intent for each person. Return JSON only.
Use pose, subpose, variant, attributes, role, and position.
Do not generate keypoints.
```
