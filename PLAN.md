# Plan: local video generator on a 4 GB GPU

This is the working plan for the video generator on an ASUS TUF Gaming F15 (Intel i5-11400H, 16 GB RAM, NVIDIA RTX 2050 4 GB). The public repository holds the window and this plan. The model weights stay on the computer.

## Goal

Generate a short mp4 on this laptop, with no cloud video API. The person using it should see one page: a prompt, an optional start picture, an optional end picture, and a saved video. They should not have to build a node graph.

## Why this stack

| Piece | Role | Why it is here |
| --- | --- | --- |
| Gradio page (`app.py`) | The window on port 7860 | Prompt, pictures, size, length, and the saved mp4 |
| ComfyUI | Hidden engine on port 8188 | Runs the LTX-2.3 graph and writes the file |
| ComfyUI-GGUF | Loads the compressed model | Streams weights through 4 GB of VRAM |
| LTX-2.3 distilled Q3_K_S | The video model | About 9.3 GB, 8 steps, the size that fits 16 GB of RAM |
| Gemma 3 12B Q3_K_S | The text encoder | Turns the prompt into the conditioning the video model reads |

A normal PyTorch load puts the whole model on the GPU. LTX-2.3 is about 18 GB of files. The RTX 2050 has 4 GB, so that load crashes. ComfyUI-GGUF keeps the model in system RAM and copies one block at a time onto the GPU. Diffusers support for LTX-2.3 was not ready when this was built, and the full safetensors file is about 44 GB.

ComfyUI is not a second product the user has to open. `app.py` starts it if port 8188 is closed, sends the graph, and waits for the mp4.

## What fits this laptop

Memory will sit near 100% while a clip runs. That is the model living in the 16 GB of RAM. The NVIDIA chip does the math. The Intel graphics only drive the screen.

Rules that keep a clip from falling apart:

- Close browsers and chat apps before Generate. If Windows starts using the disk as extra RAM, the picture drifts off the prompt and the engine can crash.
- Stay on the charger.
- Use **Fast 512×320**, **2 seconds**, and **8 steps**.
- 4 seconds, or 4 steps, on a full memory machine is the combination that already failed.
- Do not pass `--fast-disk` to ComfyUI. On this PC it crashed inside the GGUF loader.
- Use PyTorch built for CUDA 13 (`torch 2.14.1+cu130`). The CUDA 12.8 build disabled the CUDA backend on this driver and then crashed.

Width and height must be divisible by 32. Frame count must be `8k+1` (25 frames for 1 second, 49 for 2 seconds, at 24 fps).

## Repository versus the laptop

GitHub stores the window and this plan. It does not store the model, ComfyUI, or the virtual environment.

| On GitHub | On the laptop only |
| --- | --- |
| `app.py` | `ComfyUI/` including about 18 GB of weights |
| `start.bat` | `.venv/` including PyTorch |
| `README.md`, `PLAN.md` | `output/` videos |

After this plan was published, those local-only folders were removed from the laptop to free the drive. The steps below put them back.

## Restore the local install

From `D:\video-generator`, with Python 3.13 and Git installed:

```bat
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
git clone --depth 1 https://github.com/Comfy-Org/ComfyUI.git ComfyUI
.venv\Scripts\python.exe -m pip install -r ComfyUI\requirements.txt
git clone --depth 1 https://github.com/city96/ComfyUI-GGUF.git ComfyUI\custom_nodes\ComfyUI-GGUF
.venv\Scripts\python.exe -m pip install -r ComfyUI\custom_nodes\ComfyUI-GGUF\requirements.txt
.venv\Scripts\python.exe -m pip install gradio requests
.venv\Scripts\python.exe -m pip install torch==2.14.1+cu130 torchvision==0.29.1+cu130 --index-url https://download.pytorch.org/whl/cu130
```

The last command replaces whatever PyTorch ComfyUI pulled in. Confirm it with:

```bat
.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Expected: `2.14.1+cu130 True`.

## Model files

Download these five files. The names and folders matter. The audio VAE goes in `checkpoints` because the loader reads that folder, not `vae`.

| File | Size | Save as |
| --- | --- | --- |
| [distilled Q3_K_S](https://huggingface.co/unsloth/LTX-2.3-GGUF/resolve/main/distilled/ltx-2.3-22b-distilled-Q3_K_S.gguf) | 9.3 GB | `ComfyUI\models\unet\ltx-2.3-22b-distilled-Q3_K_S.gguf` |
| [Gemma Q3_K_S](https://huggingface.co/unsloth/gemma-3-12b-it-qat-GGUF/resolve/main/gemma-3-12b-it-qat-Q3_K_S.gguf) | 5.1 GB | `ComfyUI\models\text_encoders\gemma-3-12b-it-qat-Q3_K_S.gguf` |
| [embeddings connector](https://huggingface.co/unsloth/LTX-2.3-GGUF/resolve/main/text_encoders/ltx-2.3-22b-distilled_embeddings_connectors.safetensors) | 2.2 GB | `ComfyUI\models\text_encoders\ltx-2.3-22b-distilled_embeddings_connectors.safetensors` |
| [video VAE](https://huggingface.co/unsloth/LTX-2.3-GGUF/resolve/main/vae/ltx-2.3-22b-distilled_video_vae.safetensors) | 1.4 GB | `ComfyUI\models\vae\ltx-2.3-22b-distilled_video_vae.safetensors` |
| [audio VAE](https://huggingface.co/unsloth/LTX-2.3-GGUF/resolve/main/vae/ltx-2.3-22b-distilled_audio_vae.safetensors) | 0.3 GB | `ComfyUI\models\checkpoints\ltx-2.3-22b-distilled_audio_vae.safetensors` |

Q4 would follow prompts more closely and does not fit this RAM budget. The vision projector (`mmproj-BF16.gguf`) is not required for text-to-video. A log line about a missing vision tower is expected.

## How a clip is built

`app.py` sends this graph to ComfyUI:

1. Load the GGUF video model and the two text encoders.
2. Load the video VAE and the audio VAE.
3. Encode the prompt. The negative prompt is empty. Distilled LTX uses CFG 1.
4. Make an empty video latent and an empty audio latent, then join them.
5. If a start picture is uploaded, pin it at frame 0 with `LTXVAddGuide`.
6. If an end picture is uploaded, pin it at the last frame (`frame_idx` -1).
7. Sample with Euler for the chosen step count.
8. Crop the guide frames back off, decode picture and sound, and save `output\video\ltx_*.mp4`.

The page starts ComfyUI as:

```text
python main.py --listen 127.0.0.1 --port 8188 --disable-auto-launch --vram-headroom 0.5 --output-directory D:\video-generator\output
```

## How to use the page

1. Double-click `start.bat`, or run `.venv\Scripts\python.exe app.py`.
2. Open `http://127.0.0.1:7860`.
3. The model list should say `ready` on all five files.
4. Write one clear sentence. For a move from one picture to another, upload both frames.
5. Leave Fast, 2 seconds, and 8 steps.
6. Press Generate and wait. The first clip after a restart is the slow one, because the model has to load.

A 1 second, 512×320, 4 step test on this laptop finished in about 4 minutes and produced a real mp4 with AAC audio. Longer clips take much longer.

## Failures already hit

- **Memory at 98% and disk at 100%.** Other apps were open during a 4 second clip. Windows paged to disk. Close them first.
- **Engine exit code `3221225477`.** That is a Windows access violation. It happened once with CUDA 12.8 plus `--fast-disk`, and again when memory was full. The page now says the engine stopped because memory filled up, instead of printing a connection traceback.
- **`clip missing` / vision keys.** Gemma has no image tower in this download. Text-to-video still runs.
- **Prompt does not match the words.** Q3 is a compressed copy. The start and end pictures control the look more reliably than a long prompt.

## Out of scope

- Cloud rendering, batch queues, and accounts
- The full 44 GB safetensors model
- Merging disks or removing Windows recovery partitions
- Training or fine-tuning
