import argparse
import io
import json
import os
import subprocess
import tempfile
import time
from typing import Optional

import gradio as gr
import numpy as np
import torch
import soundfile as sf
from loguru import logger
from PIL import Image
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

from qwen_omni_utils import process_mm_info


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


def _save_audio_to_wav(audio: np.ndarray, sample_rate: int) -> str:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    wav_io = io.BytesIO()
    sf.write(wav_io, audio, samplerate=sample_rate, format="WAV")
    wav_io.seek(0)
    tmp_dir = tempfile.mkdtemp(prefix="thinker_audio_")
    audio_path = os.path.join(tmp_dir, "audio.wav")
    with open(audio_path, "wb") as f:
        f.write(wav_io.read())
    return audio_path


def _enqueue_job(
    queue_dir: str,
    ckpt_dir: str,
    wav2vec_dir: str,
    cond_image_path: str,
    audio_path: str,
    input_prompt: str,
    audio_encode_mode: str,
    base_seed: int,
):
    jobs_dir = os.path.join(queue_dir, "jobs")
    results_dir = os.path.join(queue_dir, "results")
    os.makedirs(jobs_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    job_id = f"job_{int(time.time() * 1000)}_{os.getpid()}"
    output_dir = os.path.join(results_dir, job_id)
    os.makedirs(output_dir, exist_ok=True)
    payload = {
        "cmd": "start",
        "ckpt_dir": ckpt_dir,
        "wav2vec_dir": wav2vec_dir,
        "cond_image_path": cond_image_path,
        "audio_path": audio_path,
        "input_prompt": input_prompt,
        "audio_encode_mode": audio_encode_mode,
        "base_seed": base_seed,
        "output_dir": output_dir,
    }
    tmp_path = os.path.join(jobs_dir, f"{job_id}.json.tmp")
    job_path = os.path.join(jobs_dir, f"{job_id}.json")
    with open(tmp_path, "w") as f:
        json.dump(payload, f)
    os.replace(tmp_path, job_path)
    return output_dir


def _launch_flashtalk_worker(
    queue_dir: str,
    gpu_ids: str,
):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_ids
    cmd = [
        "python",
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={len(gpu_ids.split(','))}",
        "src/avatar/flash_talk_queue_worker.py",
        "--queue_dir",
        queue_dir,
    ]
    logger.info("Launching FlashTalk worker: {}", " ".join(cmd))
    return subprocess.Popen(cmd, env=env)


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


def _stream_frames(output_dir: str, poll_interval: float = 0.1):
    frame_dir = os.path.join(output_dir, "frames")
    done_path = os.path.join(output_dir, "DONE")
    frame_idx = 0
    while True:
        frame_path = os.path.join(frame_dir, f"frame_{frame_idx:06d}.jpg")
        if os.path.exists(frame_path):
            yield Image.open(frame_path)
            frame_idx += 1
            continue
        if os.path.exists(done_path):
            return
        time.sleep(poll_interval)


def build_demo(
    model,
    processor,
    system_prompt: str,
    voice: str,
    use_audio_in_video: bool,
    flashtalk_ckpt_dir: str,
    flashtalk_wav2vec_dir: str,
    flashtalk_cond_image: str,
    flashtalk_prompt: str,
    flashtalk_audio_mode: str,
    flashtalk_seed: int,
    queue_dir: str,
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
            audio_path = _save_audio_to_wav(response_audio, 24000)
            output_dir = _enqueue_job(
                queue_dir,
                flashtalk_ckpt_dir,
                flashtalk_wav2vec_dir,
                flashtalk_cond_image,
                audio_path,
                flashtalk_prompt,
                flashtalk_audio_mode,
                flashtalk_seed,
            )
            yield history, None
            for frame in _stream_frames(output_dir):
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

    parser.add_argument("--queue_dir", type=str, default="tmp/flashtalk_queue")
    parser.add_argument("--start_flashtalk_worker", action="store_true")
    parser.add_argument("--flashtalk_gpus", type=str, default="1,2,3,4,5,6,7")

    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    flashtalk_proc: Optional[subprocess.Popen] = None
    if args.start_flashtalk_worker:
        flashtalk_proc = _launch_flashtalk_worker(
            queue_dir=args.queue_dir,
            gpu_ids=args.flashtalk_gpus,
        )
    model, processor = _load_model_processor(args.thinker_ckpt, args.thinker_device)
    demo = build_demo(
        model=model,
        processor=processor,
        system_prompt=args.system_prompt,
        voice=args.voice,
        use_audio_in_video=args.use_audio_in_video,
        flashtalk_ckpt_dir=args.flashtalk_ckpt,
        flashtalk_wav2vec_dir=args.flashtalk_wav2vec,
        flashtalk_cond_image=args.flashtalk_cond_image,
        flashtalk_prompt=args.flashtalk_prompt,
        flashtalk_audio_mode=args.flashtalk_audio_mode,
        flashtalk_seed=args.flashtalk_seed,
        queue_dir=args.queue_dir,
    )
    demo.launch(server_name=args.host, server_port=args.port)
    if flashtalk_proc is not None:
        flashtalk_proc.terminate()
