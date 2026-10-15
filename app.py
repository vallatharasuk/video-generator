"""Local text-to-video window for LTX 2.3 on a 4 GB GPU.

ComfyUI does the generation. This file is the interface: type a prompt, get an mp4.
"""

from __future__ import annotations

import atexit
import json
import random
import socket
import subprocess
import time
import uuid
from pathlib import Path

import gradio as gr
import requests

ROOT = Path(r"D:\video-generator")
COMFY = ROOT / "ComfyUI"
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
OUTPUT = ROOT / "output"
HOST = "127.0.0.1"
PORT = 8188

UNET = "ltx-2.3-22b-distilled-Q3_K_S.gguf"
GEMMA = "gemma-3-12b-it-qat-Q3_K_S.gguf"
CONNECTOR = "ltx-2.3-22b-distilled_embeddings_connectors.safetensors"
VIDEO_VAE = "ltx-2.3-22b-distilled_video_vae.safetensors"
AUDIO_VAE = "ltx-2.3-22b-distilled_audio_vae.safetensors"

MODELS = (
    (COMFY / "models" / "unet" / UNET, 9_000_000_000),
    (COMFY / "models" / "text_encoders" / GEMMA, 4_800_000_000),
    (COMFY / "models" / "text_encoders" / CONNECTOR, 2_000_000_000),
    (COMFY / "models" / "vae" / VIDEO_VAE, 1_200_000_000),
    (COMFY / "models" / "checkpoints" / AUDIO_VAE, 300_000_000),
)

SIZES = {
    "Fast 512x320": (512, 320),
    "Balanced 640x384": (640, 384),
    "Clearer 768x512": (768, 512),
}

_server: subprocess.Popen | None = None
_server_started_here = False


def missing_models() -> list[str]:
    missing = []
    for path, minimum in MODELS:
        if not path.exists() or path.stat().st_size < minimum:
            missing.append(path.name)
    return missing


def model_status() -> str:
    lines = []
    for path, minimum in MODELS:
        ready = path.exists() and path.stat().st_size >= minimum
        mark = "ready" if ready else "still downloading"
        lines.append(f"- {path.name}: {mark}")
    return "\n".join(lines)


def _port_open() -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.4)
        return sock.connect_ex((HOST, PORT)) == 0


def _stop_server() -> None:
    global _server
    if _server_started_here and _server is not None and _server.poll() is None:
        _server.terminate()
        try:
            _server.wait(timeout=15)
        except subprocess.TimeoutExpired:
            _server.kill()


atexit.register(_stop_server)


def ensure_server() -> None:
    global _server, _server_started_here
    if _port_open():
        return
    if not PYTHON.exists():
        raise gr.Error("Python environment is missing. The install on D:\\video-generator did not finish.")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    log_path = ROOT / "comfy.log"
    log_file = open(log_path, "a", encoding="utf-8")
    _server = subprocess.Popen(
        [
            str(PYTHON),
            "main.py",
            "--listen",
            HOST,
            "--port",
            str(PORT),
            "--disable-auto-launch",
            "--vram-headroom",
            "0.5",
            "--output-directory",
            str(OUTPUT),
        ],
        cwd=COMFY,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    _server_started_here = True
    deadline = time.time() + 180
    while time.time() < deadline:
        if _server.poll() is not None:
            raise gr.Error(f"The video engine stopped. See {log_path}")
        if _port_open():
            try:
                requests.get(f"http://{HOST}:{PORT}/system_stats", timeout=2).raise_for_status()
                return
            except requests.RequestException:
                pass
        time.sleep(1)
    raise gr.Error("The video engine did not start in time. See D:\\video-generator\\comfy.log")


def frames_for_seconds(seconds: float) -> int:
    target = max(9, int(round(seconds * 24)))
    steps = max(1, round((target - 1) / 8))
    return steps * 8 + 1


def upload_image(path: str) -> str:
    source = Path(path)
    name = f"{uuid.uuid4().hex}_{source.name}"
    with source.open("rb") as handle:
        response = requests.post(
            f"http://{HOST}:{PORT}/upload/image",
            files={"image": (name, handle, "application/octet-stream")},
            data={"overwrite": "true", "type": "input"},
            timeout=120,
        )
    if response.status_code != 200:
        raise gr.Error(response.text[:1000])
    return response.json()["name"]


def build_prompt(
    text: str,
    width: int,
    height: int,
    frames: int,
    steps: int,
    seed: int,
    start_name: str | None = None,
    end_name: str | None = None,
) -> dict:
    graph = {
        "1": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": UNET}},
        "2": {
            "class_type": "DualCLIPLoaderGGUF",
            "inputs": {"clip_name1": GEMMA, "clip_name2": CONNECTOR, "type": "ltxv"},
        },
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": VIDEO_VAE}},
        "4": {"class_type": "LTXVAudioVAELoader", "inputs": {"ckpt_name": AUDIO_VAE}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": text, "clip": ["2", 0]}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["2", 0]}},
        "7": {
            "class_type": "LTXVConditioning",
            "inputs": {"positive": ["5", 0], "negative": ["6", 0], "frame_rate": 24.0},
        },
        "8": {
            "class_type": "EmptyLTXVLatentVideo",
            "inputs": {"width": width, "height": height, "length": frames, "batch_size": 1},
        },
        "9": {
            "class_type": "LTXVEmptyLatentAudio",
            "inputs": {
                "frames_number": frames,
                "frame_rate": 24.0,
                "batch_size": 1,
                "audio_vae": ["4", 0],
            },
        },
        "10": {
            "class_type": "LTXVConcatAVLatent",
            "inputs": {"video_latent": ["8", 0], "audio_latent": ["9", 0]},
        },
        "11": {
            "class_type": "ModelSamplingLTXV",
            "inputs": {
                "model": ["1", 0],
                "max_shift": 2.05,
                "base_shift": 0.95,
                "latent": ["10", 0],
            },
        },
        "12": {
            "class_type": "LTXVScheduler",
            "inputs": {
                "steps": steps,
                "max_shift": 2.05,
                "base_shift": 0.95,
                "stretch": True,
                "terminal": 0.1,
                "latent": ["10", 0],
            },
        },
        "13": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "14": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "15": {
            "class_type": "CFGGuider",
            "inputs": {
                "model": ["11", 0],
                "positive": ["7", 0],
                "negative": ["7", 1],
                "cfg": 1.0,
            },
        },
        "16": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["14", 0],
                "guider": ["15", 0],
                "sampler": ["13", 0],
                "sigmas": ["12", 0],
                "latent_image": ["10", 0],
            },
        },
        "17": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["16", 0]}},
        "18": {"class_type": "VAEDecode", "inputs": {"samples": ["17", 0], "vae": ["3", 0]}},
        "19": {
            "class_type": "LTXVAudioVAEDecode",
            "inputs": {"samples": ["17", 1], "audio_vae": ["4", 0]},
        },
        "20": {
            "class_type": "CreateVideo",
            "inputs": {"images": ["18", 0], "fps": 24.0, "audio": ["19", 0]},
        },
        "21": {
            "class_type": "SaveVideo",
            "inputs": {"video": ["20", 0], "filename_prefix": "video/ltx", "format": "mp4"},
        },
    }
    positive, negative, video = ["7", 0], ["7", 1], ["8", 0]
    guides = False
    if start_name:
        graph["30"] = {"class_type": "LoadImage", "inputs": {"image": start_name}}
        graph["31"] = {
            "class_type": "LTXVPreprocess",
            "inputs": {"image": ["30", 0], "img_compression": 18},
        }
        graph["32"] = {
            "class_type": "LTXVAddGuide",
            "inputs": {
                "positive": positive,
                "negative": negative,
                "vae": ["3", 0],
                "latent": video,
                "image": ["31", 0],
                "frame_idx": 0,
                "strength": 1.0,
            },
        }
        positive, negative, video = ["32", 0], ["32", 1], ["32", 2]
        guides = True
    if end_name:
        graph["33"] = {"class_type": "LoadImage", "inputs": {"image": end_name}}
        graph["34"] = {
            "class_type": "LTXVPreprocess",
            "inputs": {"image": ["33", 0], "img_compression": 18},
        }
        graph["35"] = {
            "class_type": "LTXVAddGuide",
            "inputs": {
                "positive": positive,
                "negative": negative,
                "vae": ["3", 0],
                "latent": video,
                "image": ["34", 0],
                "frame_idx": -1,
                "strength": 1.0,
            },
        }
        positive, negative, video = ["35", 0], ["35", 1], ["35", 2]
        guides = True
    graph["10"]["inputs"]["video_latent"] = video
    graph["15"]["inputs"]["positive"] = positive
    graph["15"]["inputs"]["negative"] = negative
    if guides:
        graph["22"] = {
            "class_type": "LTXVCropGuides",
            "inputs": {"positive": positive, "negative": negative, "latent": ["17", 0]},
        }
        graph["18"]["inputs"]["samples"] = ["22", 2]
    return graph


def _find_video(entry: dict) -> Path | None:
    outputs = entry.get("outputs", {})
    for node in outputs.values():
        for key in ("videos", "images", "gifs"):
            for item in node.get(key, []) or []:
                name = item.get("filename", "")
                if name.lower().endswith((".mp4", ".webm", ".mkv")):
                    folder = OUTPUT / item.get("subfolder", "")
                    path = folder / name
                    if path.exists():
                        return path
    matches = sorted(OUTPUT.rglob("*.mp4"), key=lambda path: path.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def generate(
    prompt: str,
    start_image,
    end_image,
    size_label: str,
    seconds: float,
    steps: int,
    seed: int,
    progress=gr.Progress(),
):
    text = (prompt or "").strip()
    if not text and not start_image and not end_image:
        raise gr.Error("Type a prompt, or add a start frame and an end frame.")
    if not text:
        text = "Smooth cinematic motion from the first picture to the last picture."
    missing = missing_models()
    if missing:
        raise gr.Error("Models are not on disk yet: " + ", ".join(missing))

    progress(0.02, desc="Starting the video engine")
    ensure_server()
    width, height = SIZES[size_label]
    frames = frames_for_seconds(seconds)
    if seed < 0:
        seed = random.randint(0, 2**31 - 1)

    start_name = upload_image(start_image) if start_image else None
    end_name = upload_image(end_image) if end_image else None
    progress(0.08, desc="Sending the prompt")
    response = requests.post(
        f"http://{HOST}:{PORT}/prompt",
        json={
            "prompt": build_prompt(
                text, width, height, frames, int(steps), int(seed), start_name, end_name
            ),
            "client_id": str(uuid.uuid4()),
        },
        timeout=60,
    )
    if response.status_code != 200:
        detail = response.text[:2000]
        raise gr.Error(detail)
    body = response.json()
    if body.get("node_errors"):
        raise gr.Error(json.dumps(body["node_errors"])[:2000])
    prompt_id = body["prompt_id"]

    deadline = time.time() + 3 * 60 * 60
    while time.time() < deadline:
        if _server_started_here and _server is not None and _server.poll() is not None:
            raise gr.Error("The video engine stopped because memory filled up. Close other apps, then press Generate again.")
        try:
            history = requests.get(f"http://{HOST}:{PORT}/history/{prompt_id}", timeout=30).json()
        except requests.RequestException:
            raise gr.Error("The video engine stopped because memory filled up. Close other apps, then press Generate again.")
        entry = history.get(prompt_id)
        if entry:
            status = entry.get("status", {}).get("status_str", "")
            if status == "error":
                messages = entry.get("status", {}).get("messages", [])
                raise gr.Error(str(messages)[:2000])
            if status == "success":
                video = _find_video(entry)
                if video is None:
                    raise gr.Error("Generation finished but no video file was saved.")
                progress(1.0, desc="Done")
                return str(video), f"Saved {video.name}  ·  {width}x{height}  ·  {frames} frames  ·  seed {seed}"
        try:
            queue = requests.get(f"http://{HOST}:{PORT}/queue", timeout=30).json()
        except requests.RequestException:
            raise gr.Error("The video engine stopped because memory filled up. Close other apps, then press Generate again.")
        running = len(queue.get("queue_running", []))
        pending = len(queue.get("queue_pending", []))
        progress(0.2, desc=f"Generating. Close other apps. Queue running {running}, waiting {pending}")
        time.sleep(5)
    raise gr.Error("Generation ran longer than 3 hours and was stopped in the window. The engine may still be working.")


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Video Generator") as demo:
        gr.Markdown(
            """# Video Generator
Memory stays near full while a clip runs. The model is about 18 GB and the graphics card is 4 GB, so Windows holds the model in the 16 GB of RAM and only sends a slice to the GPU. Close Chrome, WhatsApp, and other apps first, or the disk starts paging and the picture drifts off the prompt.

This is the small compressed model that fits this laptop, so text alone is a hint. Upload a start frame and an end frame when the clip should animate from one picture to the other. Use **2 seconds** and **8 steps**."""
        )
        status = gr.Textbox(label="Model files", value=model_status(), lines=6, interactive=False)
        refresh = gr.Button("Check model files")
        refresh.click(model_status, outputs=status)
        prompt = gr.Textbox(
            label="Prompt",
            lines=4,
            placeholder="A red paper boat floating down a rainy street at night, cinematic, gentle camera move",
        )
        with gr.Row():
            start_image = gr.Image(
                label="Start frame",
                type="filepath",
                sources=["upload", "clipboard"],
                height=220,
            )
            end_image = gr.Image(
                label="End frame",
                type="filepath",
                sources=["upload", "clipboard"],
                height=220,
            )
        with gr.Row():
            size = gr.Dropdown(list(SIZES), value="Fast 512x320", label="Size")
            seconds = gr.Slider(1, 4, value=2, step=1, label="Seconds")
            steps = gr.Slider(4, 8, value=8, step=1, label="Steps")
            seed = gr.Number(value=-1, label="Seed (-1 = random)", precision=0)
        go = gr.Button("Generate", variant="primary")
        note = gr.Textbox(label="Result", interactive=False)
        video = gr.Video(label="Video")
        go.click(
            generate,
            [prompt, start_image, end_image, size, seconds, steps, seed],
            [video, note],
        )
    return demo


if __name__ == "__main__":
    build_ui().launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)
