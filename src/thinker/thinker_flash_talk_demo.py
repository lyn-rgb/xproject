import argparse
import os
import sys
import time
from typing import Optional

import gradio as gr
import numpy as np
import torch
from loguru import logger
from PIL import Image
from torch.multiprocessing import get_context as mp_get_context
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

from qwen_omni_utils import process_mm_info


_AVATAR_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "avatar"))
if _AVATAR_ROOT not in sys.path:
    sys.path.append(_AVATAR_ROOT)

from flash_talk_mp_worker import worker_loop as flashtalk_worker_loop


def _load_model_processor(checkpoint_path: str, device: str):
    device_map = None
    if device == "cpu":
        device_map = "cpu"
    elif device.startswith("cuda"):
        device_map = {"": int(device.split(":")[-1])}
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        checkpoint_path,
        device_map=device_map,
        torch_dtype="auto",
    )
    processor = Qwen2_5OmniProcessor.from_pretrained(checkpoint_path)
    return model, processor


def _enqueue_job(
    job_queue,
    job_id: str,
    cond_image_path: str,
    input_prompt: str,
    audio_encode_mode: str,
    base_seed: int,
    audio_array: np.ndarray,
    sample_rate: int,
):
    payload = {
        "cmd": "start",
        "job_id": job_id,
        "cond_image_path": cond_image_path,
        "input_prompt": input_prompt,
        "audio_encode_mode": audio_encode_mode,
        "base_seed": base_seed,
        "audio_array": audio_array,
        "sample_rate": sample_rate,
    }
    job_queue.put(payload)


def _launch_flashtalk_worker(
    ckpt_dir: str,
    wav2vec_dir: str,
    gpu_ids: str,
    job_queue,
    result_queue,
    init_method: str,
):
    ctx = mp_get_context("spawn")
    world_size = len(gpu_ids.split(","))
    processes = []
    for rank in range(world_size):
        proc = ctx.Process(
            target=flashtalk_worker_loop,
            args=(
                rank,
                world_size,
                init_method,
                job_queue,
                result_queue,
                ckpt_dir,
                wav2vec_dir,
                gpu_ids,
            ),
        )
        proc.start()
        processes.append(proc)
    return processes


def _format_history(history, system_prompt: str):
    messages = [{"role": "system", "content": [{"type": "text", "text": system_prompt}]}]
    for item in history:
        if isinstance(item["content"], str):
            messages.append({"role": item["role"], "content": item["content"]})
        elif item["role"] == "user" and isinstance(item["content"], (list, tuple)):
            file_path = item["content"][0]
            messages.append({"role": item["role"], "content": [{"type": "audio", "audio": file_path}]})
    return messages


def _predict(model, processor, messages, voice: str, use_audio_in_video: bool):
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(messages, use_audio_in_video=use_audio_in_video)
    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=use_audio_in_video,
    )
    inputs = inputs.to(model.device).to(model.dtype)
    text_ids, audio = model.generate(**inputs, speaker=voice, use_audio_in_video=use_audio_in_video)
    response = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    response = response[0].split("\n")[-1]
    return response, np.array(audio, dtype=np.float32)


def _stream_frames(result_queue, job_id: str, poll_interval: float = 0.1):
    while True:
        try:
            payload = result_queue.get(timeout=poll_interval)
        except Exception:
            continue
        if payload.get("job_id") != job_id:
            continue
        if payload.get("type") == "frame":
            frame = payload.get("frame")
            if frame is not None:
                yield Image.fromarray(frame)
        elif payload.get("type") == "done":
            return


def build_demo(
    model,
    processor,
    system_prompt: str,
    voice: str,
    use_audio_in_video: bool,
    flashtalk_cond_image: str,
    flashtalk_prompt: str,
    flashtalk_audio_mode: str,
    flashtalk_seed: int,
    job_queue,
    result_queue,
):
    with gr.Blocks(title="Thinker + FlashTalk") as demo:
        gr.Markdown("# Thinker Chat + FlashTalk Streaming Avatar")
        with gr.Row():
            with gr.Column(scale=1):
                system_prompt_box = gr.Textbox(label="System prompt", value=system_prompt)
                voice_box = gr.Dropdown(label="Voice", choices=["Chelsie", "Ethan"], value=voice)
                user_text = gr.Textbox(label="User text")
                user_audio = gr.Audio(label="User audio", type="filepath")
                send_btn = gr.Button("Send")
                chat_box = gr.Chatbot(label="Chat")
            with gr.Column(scale=1):
                stream_frame = gr.Image(
                    label="Live frame",
                    streaming=True,
                    show_label=True,
                    type="pil",
                    image_mode="RGB",
                )

        state = gr.State([])

        def on_send(text, audio, history, system_prompt_value, voice_value):
            if text:
                history.append({"role": "user", "content": text})
            if audio:
                history.append({"role": "user", "content": (audio,)})

            formatted = _format_history(history, system_prompt_value)
            response_text, response_audio = _predict(
                model,
                processor,
                formatted,
                voice_value,
                use_audio_in_video,
            )
            history.append({"role": "assistant", "content": response_text})
            job_id = f"job_{int(time.time() * 1000)}_{os.getpid()}"
            _enqueue_job(
                job_queue,
                job_id,
                flashtalk_cond_image,
                flashtalk_prompt,
                flashtalk_audio_mode,
                flashtalk_seed,
                response_audio,
                24000,
            )
            yield history, None
            for frame in _stream_frames(result_queue, job_id):
                yield history, frame

        send_btn.click(
            on_send,
            inputs=[user_text, user_audio, state, system_prompt_box, voice_box],
            outputs=[chat_box, stream_frame],
        )

    return demo


def _parse_args():
    parser = argparse.ArgumentParser(description="Thinker + FlashTalk merged demo.")
    parser.add_argument("--thinker_ckpt", type=str, default="models/Qwen2.5-Omni-7B")
    parser.add_argument("--thinker_device", type=str, default="cuda:0")
    parser.add_argument("--system_prompt", type=str, default=(
        "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
        "capable of perceiving auditory and visual inputs, as well as generating text and speech."
    ))
    parser.add_argument("--voice", type=str, default="Chelsie")
    parser.add_argument("--use_audio_in_video", action="store_true", default=True)

    parser.add_argument("--flashtalk_ckpt", type=str, default="models/SoulX-FlashTalk-14B")
    parser.add_argument("--flashtalk_wav2vec", type=str, default="models/chinese-wav2vec2-base")
    parser.add_argument("--flashtalk_cond_image", type=str, default="src/avatar/examples/man.png")
    parser.add_argument("--flashtalk_prompt", type=str, default="A person is talking.")
    parser.add_argument("--flashtalk_audio_mode", type=str, default="stream")
    parser.add_argument("--flashtalk_seed", type=int, default=9999)

    parser.add_argument("--start_flashtalk_worker", action="store_true")
    parser.add_argument("--flashtalk_dist_port", type=int, default=29501)
    parser.add_argument("--flashtalk_gpus", type=str, default="1,2,3,4,5,6,7")

    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    return parser.parse_args()


def _parse_gpu_index(device: str) -> Optional[int]:
    if device.startswith("cuda:"):
        try:
            return int(device.split(":")[-1])
        except ValueError:
            return None
    return None


def _assert_gpu_partition(thinker_device: str, flashtalk_gpus: str):
    thinker_idx = _parse_gpu_index(thinker_device)
    if thinker_idx is None:
        return
    flashtalk_set = {int(x) for x in flashtalk_gpus.split(",") if x.strip().isdigit()}
    if thinker_idx in flashtalk_set:
        raise ValueError(
            f"Thinker device cuda:{thinker_idx} overlaps with FlashTalk GPUs {sorted(flashtalk_set)}"
        )


if __name__ == "__main__":
    args = _parse_args()
    _assert_gpu_partition(args.thinker_device, args.flashtalk_gpus)
    flashtalk_procs = []
    job_queue = None
    result_queue = None
    if args.start_flashtalk_worker:
        ctx = mp_get_context("spawn")
        job_queue = ctx.Queue()
        result_queue = ctx.Queue()
        init_method = f"tcp://127.0.0.1:{args.flashtalk_dist_port}"
        flashtalk_procs = _launch_flashtalk_worker(
            ckpt_dir=args.flashtalk_ckpt,
            wav2vec_dir=args.flashtalk_wav2vec,
            gpu_ids=args.flashtalk_gpus,
            job_queue=job_queue,
            result_queue=result_queue,
            init_method=init_method,
        )
    else:
        raise RuntimeError("FlashTalk worker must be started in-process for direct IPC.")
    model, processor = _load_model_processor(args.thinker_ckpt, args.thinker_device)
    demo = build_demo(
        model=model,
        processor=processor,
        system_prompt=args.system_prompt,
        voice=args.voice,
        use_audio_in_video=args.use_audio_in_video,
        flashtalk_cond_image=args.flashtalk_cond_image,
        flashtalk_prompt=args.flashtalk_prompt,
        flashtalk_audio_mode=args.flashtalk_audio_mode,
        flashtalk_seed=args.flashtalk_seed,
        job_queue=job_queue,
        result_queue=result_queue,
    )
    demo.launch(server_name=args.host, server_port=args.port)
    if job_queue is not None:
        job_queue.put({"cmd": "shutdown"})
    for proc in flashtalk_procs:
        proc.terminate()
