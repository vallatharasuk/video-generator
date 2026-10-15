# Video Generator

A local text-to-video window for a laptop with a 4 GB NVIDIA GPU and 16 GB of RAM. Type a prompt, optionally upload a start frame and an end frame, and save an mp4.

The window is a small Python page. Generation runs through [ComfyUI](https://github.com/Comfy-Org/ComfyUI) and [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF), which stream [LTX-2.3](https://huggingface.co/Lightricks/LTX-2.3) through the graphics card a piece at a time. The model weights are not in this repository. They are about 18 GB and are downloaded onto the machine.

## What you get

- A page at `http://127.0.0.1:7860`
- Text-to-video with sound
- Start frame and end frame, so a clip can animate from one picture to another
- Three sizes: Fast 512×320, Balanced 640×384, Clearer 768×512
- Clip length of 1 to 4 seconds, at 24 fps

## Run it

On the machine where the models are already installed:

```bat
start.bat
```

That opens the page in the browser. Close other apps and keep the charger plugged in. Use **2 seconds** and **8 steps**. A short clip takes several minutes.

## Install it again

The model files, ComfyUI, and the Python environment are local only. [PLAN.md](PLAN.md) is the full setup: hardware limits, exact download links, folder layout, and the failures already seen on an RTX 2050.

## License

The window in this repository is the interface. LTX-2.3 weights stay under the [LTX-2 Community License](https://github.com/Lightricks/LTX-2/blob/main/LICENSE).
