import os
import socket
import subprocess
import sys
from pathlib import Path

from .nodes.ollama_pose_parser_node import OllamaPoseParserNode
from .nodes.pose_browser_node import PoseBrowserLauncherNode, PoseLoadByIdNode
from .nodes.pose_from_structure_node import PoseFromStructureNode
from .nodes.pose_matcher_node import PoseMatcherNode
from .nodes.pose_renderer_node import PoseOpenPoseRendererNode
from .nodes.pose_selector_node import PoseSelectorNode

# Import debug_log from pose_registry
try:
    from .core.pose_registry import debug_log
except ImportError:
    def debug_log(message: str):
        print(message)

OPM_NODE_CLASS_MAPPINGS = {
    "OPM_OllamaPoseParser": OllamaPoseParserNode,
    "OPM_PoseFromStructure": PoseFromStructureNode,
    "OPM_PoseSelector": PoseSelectorNode,
    "OPM_PoseByID": PoseLoadByIdNode,
    "OPM_OpenPoseRenderer": PoseOpenPoseRendererNode,
    "OPM_PoseMatcher": PoseMatcherNode,
    "OPM_OpenPoseBrowserLauncher": PoseBrowserLauncherNode,
}

NODE_CLASS_MAPPINGS = OPM_NODE_CLASS_MAPPINGS


def _register_comfy_routes() -> bool:
    try:
        from aiohttp import web
        from server import PromptServer
        from . import pose_browser_server as browser
    except Exception as exc:
        debug_log(f"[OPM OpenPose Browser] ComfyUI route registration skipped: {exc}")
        return False

    routes = PromptServer.instance.routes
    static_index = Path(__file__).parent / "web" / "pose_browser" / "index.html"

    @routes.get("/poses")
    async def pal_pose_browser_index(request):
        if static_index.exists():
            return web.FileResponse(path=str(static_index))
        return web.json_response({"error": "Pose browser index not found"}, status=404)

    @routes.get("/poses/")
    async def pal_pose_browser_index_slash(request):
        return await pal_pose_browser_index(request)

    @routes.get("/poses/api/options")
    async def pal_pose_browser_options(request):
        params = request.rel_url.query
        data = browser.get_filter_options(
            pose=params.get("pose") or None,
            gender=params.get("gender") or None,
            variant=params.get("variant") or None,
        )
        return web.json_response(data)

    @routes.get("/poses/api/filter")
    async def pal_pose_browser_filter(request):
        params = request.rel_url.query
        data = browser.filter_poses(
            pose=params.get("pose") or None,
            gender=params.get("gender") or None,
            variant=params.get("variant") or None,
            subpose=params.get("subpose") or None,
            search=params.get("search") or None,
            page=int(params.get("page") or 1),
            limit=int(params.get("limit") or 50),
        )
        return web.json_response(data)

    @routes.get("/poses/api/pose/{pose_id}")
    async def pal_pose_browser_pose(request):
        pose_id = int(request.match_info["pose_id"])
        pose_data = browser.registry.get_pose_by_id(pose_id)
        if not pose_data:
            return web.json_response({"detail": "Pose not found"}, status=404)
        return web.json_response({key: value for key, value in pose_data.items() if key != "keypoints"})

    @routes.get("/poses/api/pose/{pose_id}/files")
    async def pal_pose_browser_pose_files(request):
        pose_id = int(request.match_info["pose_id"])
        try:
            return web.json_response(browser.get_pose_files(pose_id))
        except Exception:
            return web.json_response({"detail": "Pose not found"}, status=404)

    @routes.get("/poses/api/pose/{pose_id}/copy")
    async def pal_pose_browser_pose_copy(request):
        pose_id = int(request.match_info["pose_id"])
        try:
            return web.json_response(browser.get_pose_copy_json(pose_id))
        except Exception:
            return web.json_response({"detail": "Pose JSON not found"}, status=404)

    @routes.get("/poses/api/images/{pose_id}")
    async def pal_pose_browser_image(request):
        return await pal_pose_browser_image_by_kind(request, "preview")

    @routes.get("/poses/api/images/{pose_id}/{image_kind}")
    async def pal_pose_browser_image_by_kind_route(request):
        return await pal_pose_browser_image_by_kind(request, request.match_info["image_kind"])

    async def pal_pose_browser_image_by_kind(request, image_kind):
        pose_id = int(request.match_info["pose_id"])
        pose_data = browser.registry.get_pose_by_id(pose_id)
        if not pose_data:
            return web.json_response({"detail": "Pose not found"}, status=404)
        image_path = browser._resolve_image_path(pose_data, image_kind)
        if not image_path:
            return web.json_response({"detail": "Image not found"}, status=404)
        path = Path(image_path)
        if not path.exists():
            return web.json_response({"detail": "Image file not found"}, status=404)
        return web.FileResponse(path=str(path))

    debug_log("[OPM OpenPose Browser] ComfyUI route registered at /poses")
    return True


COMFY_POSE_BROWSER_ROUTES_REGISTERED = _register_comfy_routes()


def _is_port_open(host: str, port: int) -> bool:
    test_host = "127.0.0.1" if host == "0.0.0.0" else host
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.5)
            return sock.connect_ex((test_host, port)) == 0
    except Exception:
        return False


def _launch_browser_server() -> None:
    debug_log("[DEBUG] _launch_browser_server: Called")
    host = os.getenv("OPENPOSE_BROWSER_HOST", "0.0.0.0")
    port = int(os.getenv("OPENPOSE_BROWSER_PORT", "8189"))
    debug_log(f"[DEBUG] _launch_browser_server: Checking port {host}:{port}")
    if _is_port_open(host, port):
        debug_log(f"[DEBUG] _launch_browser_server: Port {port} already open, skipping")
        return

    server_script = Path(__file__).parent / "pose_browser_server.py"
    debug_log(f"[DEBUG] _launch_browser_server: Server script path: {server_script}")
    if not server_script.exists():
        debug_log("[DEBUG] _launch_browser_server: Server script not found, skipping")
        return

    # Set environment variable for the models path
    try:
        debug_log("[DEBUG] _launch_browser_server: Trying to import folder_paths")
        import folder_paths
        models_dir = folder_paths.get_models_dir()
        env = os.environ.copy()
        env["OPENPOSE_MODELS_PATH"] = str(Path(models_dir) / "openpose")
        debug_log(f"[DEBUG] _launch_browser_server: Set OPENPOSE_MODELS_PATH to {env['OPENPOSE_MODELS_PATH']}")
    except (ImportError, AttributeError) as e:
        debug_log(f"[DEBUG] _launch_browser_server: folder_paths import failed: {e}")
        env = os.environ.copy()

    cmd = [sys.executable, str(server_script)]
    debug_log(f"[DEBUG] _launch_browser_server: Starting command: {cmd}")
    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    try:
        subprocess.Popen(
            cmd,
            cwd=str(server_script.parent),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            env=env,
            creationflags=creationflags,
        )
        debug_log(f"[DEBUG] _launch_browser_server: Server process started at http://{host}:{port}")
    except Exception as exc:
        debug_log(f"[DEBUG] _launch_browser_server: Server start failed: {exc}")


if (
    not COMFY_POSE_BROWSER_ROUTES_REGISTERED
    and os.getenv("OPENPOSE_BROWSER_AUTOSTART", "1").lower() not in {"0", "false", "no"}
):
    try:
        _launch_browser_server()
    except Exception as exc:
        debug_log(f"[OPM OpenPose Browser] auto-start initialization failed: {exc}")

WEB_DIRECTORY = "web"

OPM_NODE_DISPLAY_NAME_MAPPINGS = {
    "OPM_OllamaPoseParser": "OPM Ollama Pose Parser",
    "OPM_PoseFromStructure": "OPM Pose From Structure",
    "OPM_PoseSelector": "OPM Pose Selector",
    "OPM_PoseByID": "OPM Pose By ID",
    "OPM_OpenPoseRenderer": "OPM OpenPose Renderer",
    "OPM_PoseMatcher": "OPM Pose Matcher",
    "OPM_OpenPoseBrowserLauncher": "OPM OpenPose Browser Launcher",
}

NODE_DISPLAY_NAME_MAPPINGS = OPM_NODE_DISPLAY_NAME_MAPPINGS
