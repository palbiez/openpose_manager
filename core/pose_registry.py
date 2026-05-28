"""
Pose Registry: Central database for all poses with unique IDs.
Loads all poses from PNG-based database and provides search API.
"""

import hashlib
import importlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from PIL import Image

# DEBUG logging setup
PLUGIN_ROOT = Path(__file__).resolve().parent.parent
DEBUG_LOG_FILE = PLUGIN_ROOT / "debug_log.txt"
CACHE_FILE = PLUGIN_ROOT / "pose_registry_cache.json"
CACHE_SCHEMA_VERSION = 5

POSE_FILE_SUFFIX_RE = re.compile(
    r"_(dup\d+|duplicate\d*|copy\d*|bone_structure_full|bone_structure|openposefull|openposehand|openpose|"
    r"depthhand|normalhand|cannyhand|line_art|lineart|linart|canny|depth|normal)"
    r"(_[a-z0-9]+)?$",
    re.IGNORECASE,
)

def debug_log(message: str):
    """Write debug message to log file and console."""
    timestamp = __import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    full_message = f"[{timestamp}] {message}"
    print(full_message)
    try:
        with open(DEBUG_LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(full_message + '\n')
    except Exception as e:
        print(f"[DEBUG] Failed to write to log file: {e}")

# Clear log file on import
try:
    DEBUG_LOG_FILE.unlink(missing_ok=True)
    debug_log("=== DEBUG LOG STARTED ===")
except:
    pass


class PoseRegistry:
    """Singleton registry for all available poses."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            debug_log("[DEBUG] PoseRegistry.__init__: Already initialized, skipping")
            return
        
        debug_log("[DEBUG] PoseRegistry.__init__: Starting initialization")
        self._initialized = True
        self.poses: List[Dict] = []
        self.poses_by_id: Dict[int, Dict] = {}
        self.index_by_filter: Dict[Tuple, List[int]] = {}
        
        # Try to load from cache first
        debug_log("[DEBUG] PoseRegistry.__init__: Attempting to load from cache")
        if self._load_from_cache():
            debug_log(f"[DEBUG] PoseRegistry.__init__: Successfully loaded {len(self.poses)} poses from cache")
        else:
            debug_log("[DEBUG] PoseRegistry.__init__: Cache load failed, loading poses from disk")
            self._load_poses()
            self._save_to_cache()
        
        debug_log(f"[DEBUG] PoseRegistry.__init__: Building index for {len(self.poses)} poses")
        self._build_index()
        debug_log(f"[DEBUG] PoseRegistry.__init__: Initialization complete. Total poses: {len(self.poses)}")

    def _build_source_fingerprint(self, openpose_dir: Path) -> Dict:
        """Build a compact fingerprint of pose source files for cache invalidation."""
        if not openpose_dir.exists():
            return {
                "root": str(openpose_dir),
                "exists": False,
                "file_count": 0,
                "newest_mtime": 0,
                "digest": "",
            }

        digest = hashlib.sha256()
        file_count = 0
        newest_mtime = 0

        source_files = []
        for pattern in ("*.png", "*.json"):
            source_files.extend(openpose_dir.rglob(pattern))

        for pose_path in sorted(source_files, key=lambda path: path.relative_to(openpose_dir).as_posix().lower()):
            if self._is_skipped_pose_file(pose_path):
                continue
            try:
                stat = pose_path.stat()
                rel_path = pose_path.relative_to(openpose_dir).as_posix()
            except OSError:
                continue

            digest.update(rel_path.encode("utf-8", errors="surrogatepass"))
            digest.update(b"\0")
            digest.update(str(stat.st_size).encode("ascii"))
            digest.update(b"\0")
            digest.update(str(stat.st_mtime_ns).encode("ascii"))
            digest.update(b"\0")
            file_count += 1
            newest_mtime = max(newest_mtime, stat.st_mtime)

        return {
            "root": str(openpose_dir.resolve()),
            "exists": True,
            "file_count": file_count,
            "newest_mtime": newest_mtime,
            "digest": digest.hexdigest(),
        }

    @staticmethod
    def _source_fingerprints_match(cached: object, current: Dict) -> bool:
        if not isinstance(cached, dict):
            return False
        return all(
            cached.get(key) == current.get(key)
            for key in ("root", "file_count", "digest")
        )

    @staticmethod
    def _cached_pose_source_exists(pose: Dict) -> bool:
        source_fields = (
            "json_path",
            "display_image",
            "bone_structure_path",
            "bone_structure_full_path",
            "png_path",
            "source_file",
        )
        for field in source_fields:
            value = pose.get(field)
            if value and Path(value).exists():
                return True
        return False
    
    def _load_from_cache(self) -> bool:
        """Try to load poses from cache file. Returns True if successful."""
        cache_file = CACHE_FILE
        debug_log(f"[DEBUG] _load_from_cache: Checking cache file: {cache_file}")
        if not cache_file.exists():
            debug_log("[DEBUG] _load_from_cache: Cache file does not exist")
            return False
        
        debug_log(f"[DEBUG] _load_from_cache: Cache file exists, size: {cache_file.stat().st_size} bytes")
        
        try:
            debug_log("[DEBUG] _load_from_cache: Reading cache file")
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            debug_log(f"[DEBUG] _load_from_cache: Cache data loaded, keys: {list(cache_data.keys())}")
            if cache_data.get("schema_version") != CACHE_SCHEMA_VERSION:
                debug_log("[DEBUG] _load_from_cache: Cache schema changed, rescanning...")
                return False
            
            # Check if cache is still valid by comparing the complete source set.
            # This detects additions, edits, renames, and deletions.
            openpose_dir = self._get_openpose_dir()
            debug_log(f"[DEBUG] _load_from_cache: OpenPose directory: {openpose_dir}")
            if openpose_dir.exists():
                current_fingerprint = self._build_source_fingerprint(openpose_dir)
                cached_fingerprint = cache_data.get("source_fingerprint")

                debug_log(
                    "[DEBUG] _load_from_cache: Source fingerprint "
                    f"files={current_fingerprint['file_count']} "
                    f"newest_mtime={current_fingerprint['newest_mtime']} "
                    f"digest={current_fingerprint['digest'][:12]}"
                )

                if not self._source_fingerprints_match(cached_fingerprint, current_fingerprint):
                    debug_log("[DEBUG] _load_from_cache: Source files changed, rescanning...")
                    return False
            
            # Load from cache. The cache intentionally stores only metadata and
            # source paths; keypoints are loaded lazily from the JSON/PNG files.
            debug_log("[DEBUG] _load_from_cache: Loading poses from cache data")
            self.poses = []
            self.poses_by_id = {}
            for value in cache_data.get("poses", []):
                if not isinstance(value, dict):
                    continue
                pose_copy = value.copy()
                self.poses.append(pose_copy)
                try:
                    key_int = int(pose_copy["id"])
                except Exception:
                    continue
                self.poses_by_id[key_int] = pose_copy
            
            # If cache contains no poses but OpenPose data exists, force a rescan.
            if len(self.poses) == 0 and openpose_dir.exists():
                has_pose_files = any(openpose_dir.rglob("*.png")) or any(openpose_dir.rglob("*.json"))
                if has_pose_files:
                    debug_log("[DEBUG] _load_from_cache: WARNING: cache is empty but OpenPose files are present; rescanning...")
                    return False

            missing_cached_sources = [
                pose for pose in self.poses
                if not self._cached_pose_source_exists(pose)
            ]
            if missing_cached_sources:
                debug_log(
                    "[DEBUG] _load_from_cache: "
                    f"{len(missing_cached_sources)} cached poses point to missing source files; rescanning..."
                )
                self.poses = []
                self.poses_by_id = {}
                return False
            
            debug_log("[DEBUG] _load_from_cache: Cache load successful")
            return True
            
        except Exception as e:
            debug_log(f"[DEBUG] _load_from_cache: Error loading cache: {e}")
            return False
    
    def _save_to_cache(self):
        """Save current poses to cache file."""
        try:
            cache_file = CACHE_FILE
            openpose_dir = self._get_openpose_dir()
            
            cache_fields = [
                "id",
                "pose",
                "gender",
                "variant",
                "subpose",
                "attributes",
                "base_name",
                "png_path",
                "json_path",
                "display_image",
                "bone_structure_path",
                "bone_structure_full_path",
                "source_file",
            ]

            # Convert poses to a compact serializable format. Do not persist
            # keypoints or the duplicate files list; both can be derived from
            # the source files when needed.
            serializable_poses = []
            for pose in self.poses:
                pose_copy = {field: pose.get(field) for field in cache_fields if field in pose}
                serializable_poses.append(pose_copy)

            cache_data = {
                "schema_version": CACHE_SCHEMA_VERSION,
                "source_fingerprint": self._build_source_fingerprint(openpose_dir),
                "poses": serializable_poses,
                "total_poses": len(self.poses)
            }
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
            
            print(f"[PoseRegistry] Saved cache with {len(self.poses)} poses")
            
        except Exception as e:
            print(f"[PoseRegistry] Error saving cache: {e}")
    
    def _get_openpose_dir(self) -> Path:
        """Get OpenPose directory from ComfyUI models folder."""
        env_path = os.getenv("OPENPOSE_MODELS_PATH")
        if env_path:
            candidate = Path(env_path)
            if candidate.exists():
                return candidate
            print(f"[PoseRegistry] WARNING: OPENPOSE_MODELS_PATH is set but does not exist: {candidate}")

        models_dir = self._resolve_models_dir()
        if models_dir is None:
            print("[PoseRegistry] ERROR: could not resolve ComfyUI models directory from folder_paths or package location")
            return Path("")

        openpose_dir = models_dir / "openpose"
        if openpose_dir.exists():
            return openpose_dir

        print(f"[PoseRegistry] WARNING: expected openpose directory not found: {openpose_dir}")
        return Path("")

    def _resolve_models_dir(self) -> Optional[Path]:
        models_dir = None
        try:
            folder_paths = importlib.import_module("folder_paths")
        except (ImportError, ModuleNotFoundError) as exc:
            print(f"[PoseRegistry] INFO: folder_paths module unavailable outside ComfyUI: {exc}")
            folder_paths = None

        if folder_paths is not None:
            for attr_name in ["get_models_dir", "get_models_directory", "get_models_path", "get_model_dir"]:
                if hasattr(folder_paths, attr_name):
                    try:
                        candidate = Path(getattr(folder_paths, attr_name)())
                        if candidate.exists():
                            models_dir = candidate
                            break
                    except Exception:
                        continue

            if models_dir is None and hasattr(folder_paths, "get_input_directory"):
                try:
                    input_dir = Path(folder_paths.get_input_directory())
                    candidate = input_dir.parent / "models"
                    if candidate.exists():
                        models_dir = candidate
                    else:
                        candidate = (input_dir / ".." / "models").resolve()
                        if candidate.exists():
                            models_dir = candidate
                except Exception:
                    pass

        if models_dir is None:
            models_dir = self._find_models_dir_by_search()
            if models_dir is not None:
                print(f"[PoseRegistry] INFO: Found ComfyUI models directory by search: {models_dir}")

        return models_dir

    @staticmethod
    def _find_models_dir_by_search() -> Optional[Path]:
        search_roots = [PLUGIN_ROOT, Path.cwd()]
        for root in search_roots:
            for parent in [root] + list(root.parents):
                candidate = parent / "models"
                if candidate.exists() and (candidate / "openpose").exists():
                    return candidate
                candidate = parent / "ComfyUI" / "models"
                if candidate.exists() and (candidate / "openpose").exists():
                    return candidate
        return None

    @staticmethod
    def _normalize_token(value: str) -> str:
        return str(value).strip().lower().replace(" ", "_").replace("-", "_")

    @staticmethod
    def _is_skipped_pose_file(path: Path) -> bool:
        name = path.name.lower()
        stem = path.stem.strip().lower()
        if stem.startswith(("cover", "example", "thumb")):
            return True
        if "pose_thumbnails" in [part.lower() for part in path.parts]:
            return True
        return name.endswith((".bak", ".tmp"))

    @staticmethod
    def _strip_pose_file_suffix(stem: str) -> str:
        base = stem.strip()
        while True:
            match = POSE_FILE_SUFFIX_RE.search(base)
            if not match:
                return base.strip()
            base = base[:match.start()].rstrip()

    def _derive_pose_metadata(self, file_path: Path, openpose_dir: Path, base_name: Optional[str] = None):
        relative = file_path.relative_to(openpose_dir)
        dir_parts = [self._normalize_token(p) for p in relative.parent.parts if p]

        pose = "unknown"
        gender = "unknown"
        variant = "base"
        subpose = "default"

        if len(dir_parts) >= 4:
            pose, gender, variant, subpose = dir_parts[:4]
        elif len(dir_parts) == 3:
            pose, variant, subpose = dir_parts
            gender = "unknown"
        elif len(dir_parts) == 2:
            pose, variant = dir_parts
            subpose = "default"
            gender = "unknown"
        else:
            filename = self._normalize_token(base_name or self._strip_pose_file_suffix(relative.stem))
            tokens = filename.split("_")
            if len(tokens) >= 3:
                pose, variant, subpose = tokens[:3]
            elif len(tokens) == 2:
                pose, variant = tokens
                subpose = "default"
            else:
                pose = tokens[0]

        return pose, gender, variant, subpose

    @staticmethod
    def _select_file_by_suffix(files: List[Path], base_name: str, suffix: str) -> Optional[Path]:
        expected = f"{base_name}{suffix}".lower()
        for candidate in files:
            if candidate.stem.lower() == expected:
                return candidate
        return None

    @staticmethod
    def _select_file_by_pattern(files: List[Path], base_name: str, suffix: str) -> Optional[Path]:
        prefix = f"{base_name}{suffix}_".lower()
        matches = [candidate for candidate in files if candidate.stem.lower().startswith(prefix)]
        return sorted(matches, key=lambda p: p.name.lower())[0] if matches else None

    def _find_associated_images(self, base_name: str, files: List[Path]):
        images = sorted(
            [path for path in files if path.suffix.lower() == ".png" and not self._is_skipped_pose_file(path)],
            key=lambda path: path.name.lower(),
        )
        depth = self._select_file_by_suffix(images, base_name, "_depth")
        depth_variant = self._select_file_by_pattern(images, base_name, "_depth")
        bone_structure_full = self._select_file_by_suffix(images, base_name, "_bone_structure_full")
        bone_structure = self._select_file_by_suffix(images, base_name, "_bone_structure")

        display = depth or depth_variant or bone_structure or bone_structure_full

        if bone_structure is None and bone_structure_full is not None:
            bone_structure = bone_structure_full

        return display, bone_structure, bone_structure_full

    def _find_associated_json(self, base_name: str, files: List[Path]) -> Optional[Path]:
        json_files = sorted(
            [path for path in files if path.suffix.lower() == ".json" and not self._is_skipped_pose_file(path)],
            key=lambda path: path.name.lower(),
        )
        if not json_files:
            return None

        base = base_name.lower()
        exact_openpose = f"{base}_openpose"
        exact_plain = base

        def score(candidate: Path):
            stem = candidate.stem.lower()
            if stem == exact_openpose:
                return (0, stem)
            if stem == exact_plain:
                return (1, stem)
            if stem.startswith(f"{base}_") and stem.endswith("_openpose"):
                variant = stem[len(base) + 1:-len("_openpose")]
                duplicate_penalty = 10 if variant.startswith(("dup", "duplicate", "copy")) else 0
                return (2 + duplicate_penalty, stem)
            return (99, stem)

        matches = [candidate for candidate in json_files if score(candidate)[0] < 99]
        return sorted(matches, key=score)[0] if matches else json_files[0]

    @staticmethod
    def _normalize_attributes(attributes) -> List[str]:
        if attributes is None:
            return []
        if isinstance(attributes, str):
            attributes = [attributes]
        if not isinstance(attributes, list):
            attributes = [attributes]

        normalized = []
        seen = set()
        for attribute in attributes:
            value = str(attribute).strip().lower().replace(" ", "_").replace("-", "_")
            if value and value not in seen:
                normalized.append(value)
                seen.add(value)
        return normalized

    def _extract_attributes_from_json(self, json_path: Optional[Path]) -> List[str]:
        if json_path is None:
            return []
        try:
            with open(json_path, "r", encoding="utf-8-sig") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[PoseRegistry] Error extracting attributes from {json_path}: {e}")
            return []

        if not isinstance(data, dict):
            return []

        attributes = []
        meta = data.get("meta")
        if isinstance(meta, dict):
            attributes.extend(self._normalize_attributes(meta.get("attributes")))
            attributes.extend(self._normalize_attributes(meta.get("auto_attributes")))

        attributes.extend(self._normalize_attributes(data.get("attributes")))

        people = data.get("people")
        if isinstance(people, list) and people and isinstance(people[0], dict):
            attributes.extend(self._normalize_attributes(people[0].get("attributes")))

        return self._normalize_attributes(attributes)

    def _scan_openpose_folder(self, openpose_dir: Path, pose_attributes):
        grouped_files: Dict[Tuple[str, str], List[Path]] = {}
        group_names: Dict[Tuple[str, str], str] = {}

        for file_path in openpose_dir.rglob("*"):
            if not file_path.is_file() or file_path.suffix.lower() not in {".png", ".json"}:
                continue
            if self._is_skipped_pose_file(file_path):
                continue

            base_name = self._strip_pose_file_suffix(file_path.stem)
            if not base_name:
                continue

            key = (str(file_path.parent.resolve()).lower(), base_name.lower())
            grouped_files.setdefault(key, []).append(file_path)
            group_names.setdefault(key, base_name)

        pose_id = 1

        for key in sorted(grouped_files.keys(), key=lambda value: (value[0], value[1])):
            files = grouped_files[key]
            base_name = group_names[key]
            sample_path = sorted(files, key=lambda path: path.name.lower())[0]
            pose, gender, variant, subpose = self._derive_pose_metadata(sample_path, openpose_dir, base_name)
            
            # Use filename as pose name if metadata extraction failed
            if pose == "unknown":
                pose = self._normalize_token(base_name)
            
            display_image, bone_structure, bone_structure_full = self._find_associated_images(base_name, files)
            json_path = self._find_associated_json(base_name, files)
            primary_path = display_image or bone_structure or bone_structure_full or json_path or sample_path
            file_attributes = self._extract_attributes_from_json(json_path)
            mapped_attributes = pose_attributes.get((pose, variant, subpose), [])
            attributes = self._normalize_attributes(mapped_attributes + file_attributes)
            
            pose_data = {
                "id": pose_id,
                "pose": pose,
                "gender": gender,
                "variant": variant,
                "subpose": subpose,
                "attributes": attributes,
                "base_name": base_name,
                "png_path": str(primary_path) if primary_path and primary_path.suffix.lower() == ".png" else "",
                "json_path": str(json_path) if json_path else "",
                "display_image": str(display_image) if display_image else "",
                "bone_structure_path": str(bone_structure) if bone_structure else "",
                "bone_structure_full_path": str(bone_structure_full) if bone_structure_full else "",
                "source_file": base_name,
                "files": [str(path) for path in sorted(files, key=lambda path: path.name.lower())],
            }

            self.poses.append(pose_data)
            self.poses_by_id[pose_id] = pose_data
            pose_id += 1

        print(f"[PoseRegistry] Scanned {len(self.poses)} pose groups")

    def _extract_keypoints_from_pose_file(self, json_path: Optional[Path], image_path: Optional[Path]) -> Optional[List[float]]:
        if json_path is not None:
            keypoints = self._extract_keypoints_from_json(json_path)
            if keypoints:
                return keypoints
        if image_path is not None:
            return self._extract_keypoints_from_png(image_path)
        return None

    def _extract_keypoints_from_json(self, json_path: Path) -> Optional[List[float]]:
        """Extract keypoints from OpenPose JSON or a simple keypoints payload."""
        try:
            with open(json_path, "r", encoding="utf-8-sig") as f:
                data = json.load(f)

            if isinstance(data, dict):
                if isinstance(data.get("keypoints"), list):
                    return data["keypoints"]
                people = data.get("people")
                if isinstance(people, list) and people:
                    person = people[0]
                    if isinstance(person, dict):
                        for key in ("pose_keypoints_2d", "keypoints"):
                            value = person.get(key)
                            if isinstance(value, list):
                                return value
                if isinstance(data.get("pose_keypoints_2d"), list):
                    return data["pose_keypoints_2d"]

            if isinstance(data, list):
                if data and isinstance(data[0], dict) and isinstance(data[0].get("keypoints"), list):
                    return data[0]["keypoints"]
                if all(isinstance(value, (int, float)) for value in data):
                    return data
        except Exception as e:
            print(f"[PoseRegistry] Error extracting keypoints from {json_path}: {e}")

        return None

    def _extract_keypoints_from_png(self, png_path: Path) -> Optional[List[float]]:
        """Extract keypoints from PNG file EXIF or info."""
        try:
            # Try PIL first
            with Image.open(png_path) as img:
                for key, value in img.info.items():
                    if isinstance(value, str):
                        text = value.strip()
                        if text.startswith("{") or text.startswith("["):
                            try:
                                data = json.loads(text)
                                if isinstance(data, dict) and "keypoints" in data:
                                    return data["keypoints"]
                                elif isinstance(data, list):
                                    return data
                            except json.JSONDecodeError:
                                continue

            # Fallback: try exiftool if the user configured it explicitly.
            try:
                exiftool_path = os.getenv("EXIFTOOL_PATH")
                if exiftool_path and os.path.exists(exiftool_path):
                    result = subprocess.run(
                        [exiftool_path, "-j", str(png_path)],
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        errors="ignore"
                    )

                    if result.returncode == 0 and result.stdout:
                        data = json.loads(result.stdout)[0]
                        for key, value in data.items():
                            if isinstance(value, str):
                                text = value.strip()
                                if text.startswith("{") or text.startswith("["):
                                    try:
                                        parsed = json.loads(text)
                                        if isinstance(parsed, dict) and "keypoints" in parsed:
                                            return parsed["keypoints"]
                                        elif isinstance(parsed, list):
                                            return parsed
                                    except json.JSONDecodeError:
                                        continue
            except Exception:
                pass

        except Exception as e:
            print(f"[PoseRegistry] Error extracting keypoints from {png_path}: {e}")

        return None

    def _find_alternate_png_path(self, png_path: Path) -> Optional[Path]:
        """Find a best-fit PNG alternative when the exact file is missing."""
        if png_path.exists():
            return png_path

        parent = png_path.parent
        if not parent.exists():
            return None

        candidates = [
            p for p in parent.glob("*.png")
            if p.name.lower() not in {"cover.png", "example.png"}
        ]
        if not candidates:
            return None

        subpose_name = parent.name.lower()

        def score(candidate: Path) -> int:
            name = candidate.name.lower()
            score = 0
            if subpose_name and subpose_name in name:
                score += 50
            if "_bone_structure_full" in name:
                score += 40
            elif "_bone_structure.png" in name:
                score += 30
            elif "_openposehand" in name:
                score += 25
            elif "_canny" in name:
                score += 20
            elif "_depth" in name:
                score += 15
            elif "_normalhand" in name:
                score += 10
            elif "openposefull" in name:
                score += 5
            if name == png_path.name.lower():
                score += 100
            return score

        candidates.sort(key=score, reverse=True)
        best = candidates[0]
        if score(best) > 0:
            return best
        return None

    def _load_poses(self):
        """Load all poses from PNG database."""
        openpose_dir = self._get_openpose_dir()
        print(f"[PoseRegistry] Loading from: {openpose_dir}")

        if not openpose_dir.exists():
            print(f"[PoseRegistry] WARNING: Directory does not exist: {openpose_dir}")
            return

        # Load pose mapping for attributes
        pose_mapping_path = openpose_dir / "pose_mapping.json"
        pose_attributes = {}
        if pose_mapping_path.exists():
            try:
                with open(pose_mapping_path, "r", encoding="utf-8-sig") as f:
                    pose_mapping = json.load(f)
                    for pose_name, variants in pose_mapping.items():
                        for variant_name, subposes in variants.items():
                            for subpose_name, data in subposes.items():
                                key = (pose_name, variant_name, subpose_name)
                                pose_attributes[key] = data.get("attributes", [])
            except Exception as e:
                print(f"[PoseRegistry] Error loading pose_mapping.json: {e}")

        pose_index_path = openpose_dir / "pose_index.json"
        if pose_index_path.exists():
            try:
                with open(pose_index_path, "r", encoding="utf-8-sig") as f:
                    pose_index = json.load(f)
            except Exception as e:
                print(f"[PoseRegistry] Error loading pose_index.json: {e}")
                return

            pose_id = 1
            for pose_name, genders in pose_index.items():
                for gender_name, variants in genders.items():
                    for variant_name, subposes in variants.items():
                        for subpose_name, png_files in subposes.items():
                            for png_file in png_files:
                                file_path = str(png_file).replace("\\", "/").replace("\r", "").replace("\n", "").strip()
                                png_path = openpose_dir / Path(file_path)

                                if not png_path.exists():
                                    alternate_path = self._find_alternate_png_path(png_path)
                                    if alternate_path is not None and alternate_path.exists():
                                        print(f"[PoseRegistry] Resolved missing file {png_path.relative_to(openpose_dir)} -> {alternate_path.relative_to(openpose_dir)}")
                                        png_path = alternate_path

                                if png_path.exists():
                                    base_name = self._strip_pose_file_suffix(png_path.stem)
                                    sibling_files = [
                                        path for path in png_path.parent.iterdir()
                                        if path.is_file() and path.suffix.lower() in {".png", ".json"}
                                           and self._strip_pose_file_suffix(path.stem).lower() == base_name.lower()
                                    ]
                                    display_image, bone_structure, bone_structure_full = self._find_associated_images(base_name, sibling_files)
                                    json_path = self._find_associated_json(base_name, sibling_files)
                                    attr_key = (pose_name, variant_name, subpose_name)
                                    file_attributes = self._extract_attributes_from_json(json_path)
                                    mapped_attributes = pose_attributes.get(attr_key, [])
                                    attributes = self._normalize_attributes(mapped_attributes + file_attributes)

                                    pose_data = {
                                        "id": pose_id,
                                        "pose": pose_name,
                                        "gender": gender_name,
                                        "variant": variant_name,
                                        "subpose": subpose_name,
                                        "attributes": attributes,
                                        "base_name": base_name,
                                        "png_path": str(display_image or png_path),
                                        "json_path": str(json_path) if json_path else "",
                                        "display_image": str(display_image) if display_image else str(png_path),
                                        "bone_structure_path": str(bone_structure) if bone_structure else "",
                                        "bone_structure_full_path": str(bone_structure_full) if bone_structure_full else "",
                                        "source_file": base_name,
                                        "files": [str(path) for path in sorted(sibling_files, key=lambda path: path.name.lower())],
                                    }

                                    self.poses.append(pose_data)
                                    self.poses_by_id[pose_id] = pose_data
                                    pose_id += 1
                                else:
                                    print(f"[PoseRegistry] PNG file not found: {png_path}")
            print(f"[PoseRegistry] Loaded {len(self.poses)} poses")
            return

        self._scan_openpose_folder(openpose_dir, pose_attributes)
    
    def _build_index(self):
        """Build index for faster searches."""
        for pose_data in self.poses:
            pose = pose_data["pose"]
            variant = pose_data["variant"]
            subpose = pose_data["subpose"]
            
            # Index by (pose, variant, subpose)
            key = (pose, variant, subpose)
            if key not in self.index_by_filter:
                self.index_by_filter[key] = []
            self.index_by_filter[key].append(pose_data["id"])
    
    def get_pose_by_id(self, pose_id: int) -> Optional[Dict]:
        """Get a pose by its ID."""
        return self.poses_by_id.get(pose_id)
    
    def get_keypoints_by_id(self, pose_id: int) -> Optional[List[float]]:
        """Get keypoints for a pose ID."""
        pose = self.get_pose_by_id(pose_id)
        if not pose:
            return None

        keypoints = pose.get("keypoints")
        if keypoints:
            return keypoints

        json_path = pose.get("json_path") or ""
        image_path = (
            pose.get("display_image")
            or pose.get("bone_structure_path")
            or pose.get("bone_structure_full_path")
            or pose.get("png_path")
            or ""
        )

        keypoints = self._extract_keypoints_from_pose_file(
            Path(json_path) if json_path else None,
            Path(image_path) if image_path else None,
        ) or []
        pose["keypoints"] = keypoints
        return keypoints if keypoints else None

    def get_pose_json_text_by_id(self, pose_id: int) -> Optional[str]:
        """Get copyable JSON for a pose ID, preferring the source OpenPose JSON."""
        pose = self.get_pose_by_id(pose_id)
        if not pose:
            return None

        json_path = pose.get("json_path") or ""
        if json_path:
            path = Path(json_path)
            if path.exists():
                try:
                    return path.read_text(encoding="utf-8-sig")
                except Exception as e:
                    print(f"[PoseRegistry] Error reading pose JSON from {path}: {e}")

        keypoints = self.get_keypoints_by_id(pose_id)
        if not keypoints:
            return None

        payload = {
            "pose": pose.get("pose"),
            "gender": pose.get("gender"),
            "variant": pose.get("variant"),
            "subpose": pose.get("subpose"),
            "source_file": pose.get("source_file"),
            "keypoints": keypoints,
        }
        return json.dumps(payload, indent=2, ensure_ascii=False)
    
    def search(
        self,
        pose: Optional[str] = None,
        gender: Optional[str] = None,
        variant: Optional[str] = None,
        subpose: Optional[str] = None,
    ) -> List[int]:
        """
        Search for pose IDs matching criteria.
        
        Args:
            pose: Pose name (e.g., "standing", "lying", "kneeling")
            gender: Gender folder token (e.g., "f", "m", "unknown")
            variant: Variant (e.g., "base", "nsfw")
            subpose: Subpose (e.g., "neutral", "prone", "supine")
        
        Returns:
            List of matching pose IDs
        """
        debug_log(f"[DEBUG] search: Called with pose={pose}, gender={gender}, variant={variant}, subpose={subpose}")
        debug_log(f"[DEBUG] search: Registry has {len(self.poses)} total poses")
        
        if pose is None and gender is None and variant is None and subpose is None:
            # Return all poses
            debug_log(f"[DEBUG] search: No pose filter, returning all {len(self.poses)} pose IDs")
            return [p["id"] for p in self.poses]
        
        # Get all poses matching the given criteria
        results = []
        for pose_data in self.poses:
            if pose is not None and pose_data["pose"] != pose:
                continue
            if gender is not None and pose_data.get("gender") != gender:
                continue
            if variant is not None and pose_data["variant"] != variant:
                continue
            if subpose is not None and pose_data["subpose"] != subpose:
                continue
            results.append(pose_data["id"])
        
        debug_log(f"[DEBUG] search: Found {len(results)} matching poses")
        return results
    
    def get_available_genders(self, pose: Optional[str] = None) -> List[str]:
        """Get available gender tokens, optionally scoped to a pose."""
        genders = set()
        for pose_data in self.poses:
            if pose and pose_data["pose"] != pose:
                continue
            genders.add(pose_data.get("gender") or "unknown")
        return sorted(list(genders))

    def get_available_variants(self, pose: str, gender: Optional[str] = None) -> List[str]:
        """Get available variants for a pose."""
        variants = set()
        for pose_data in self.poses:
            if pose_data["pose"] != pose:
                continue
            if gender and pose_data.get("gender") != gender:
                continue
            variants.add(pose_data["variant"])
        return sorted(list(variants))
    
    def get_available_subposes(self, pose: str, variant: str, gender: Optional[str] = None) -> List[str]:
        """Get available subposes for a pose/variant combination."""
        subposes = set()
        for pose_data in self.poses:
            if pose_data["pose"] != pose or pose_data["variant"] != variant:
                continue
            if gender and pose_data.get("gender") != gender:
                continue
            subposes.add(pose_data["subpose"])
        return sorted(list(subposes))
    
    def get_all_poses(self) -> List[str]:
        """Get list of all unique pose names."""
        poses = set()
        for pose_data in self.poses:
            poses.add(pose_data["pose"])
        return sorted(list(poses))
    
    def get_info_by_id(self, pose_id: int) -> str:
        """Get human-readable info for a pose ID."""
        pose = self.get_pose_by_id(pose_id)
        if not pose:
            return f"ID {pose_id}: NOT FOUND"
        
        return f"ID {pose_id}: {pose['pose']} / {pose['variant']} / {pose['subpose']}"
    
    def list_all(self) -> List[Dict]:
        """Get all poses with their metadata (excluding keypoints)."""
        debug_log(f"[DEBUG] list_all: Called, registry has {len(self.poses)} poses")
        result = [
            {
                "id": p["id"],
                "pose": p["pose"],
                "gender": p.get("gender"),
                "variant": p["variant"],
                "subpose": p["subpose"],
                "base_name": p.get("base_name"),
                "attributes": p.get("attributes", []),
                "source_file": p.get("source_file") or Path(p.get("png_path", "")).name,
                "has_preview": bool(p.get("display_image") or p.get("bone_structure_path") or p.get("bone_structure_full_path")),
            }
            for p in self.poses
        ]
        debug_log(f"[DEBUG] list_all: Returning {len(result)} pose summaries")
        return result


# Global singleton instance
registry = None


def get_registry() -> PoseRegistry:
    """Get the global pose registry instance."""
    global registry
    debug_log(f"[DEBUG] get_registry: Called, current registry is {'None' if registry is None else 'initialized'}")
    if registry is None:
        debug_log("[DEBUG] get_registry: Creating new PoseRegistry instance")
        registry = PoseRegistry()
        debug_log(f"[DEBUG] get_registry: Registry created with {len(registry.poses)} poses")
    else:
        debug_log(f"[DEBUG] get_registry: Returning existing registry with {len(registry.poses)} poses")
    return registry
