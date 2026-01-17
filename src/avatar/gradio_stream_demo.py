# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
import asyncio
import os
import subprocess
import tempfile
import time
from collections import deque
from datetime import datetime
import queue
import sys

import gradio as gr
import imageio
import librosa
import numpy as np
import torch
import torch.distributed as dist
from loguru import logger
from PIL import Image

from flash_talk.inference import (
    get_pipeline,
    get_base_data,
    get_audio_embedding,
    run_pipeline,
    infer_params,
)

_FRTC_BACKEND = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "3rd_party",
        "OpenAvatarChat",
        "src",
        "third_party",
        "gradio_webrtc_videochat",
        "backend",
    )
)
if _FRTC_BACKEND not in sys.path:
    sys.path.insert(0, _FRTC_BACKEND)

try:
    from fastrtc import (
        AsyncAudioVideoStreamHandler,
        WebRTC,
        VideoEmitType,
        AudioEmitType,
    )
    _RTC_AVAILABLE = True
except Exception as exc:
    logger.warning("fastrtc unavailable, WebRTC stream disabled: {}", exc)
    _RTC_AVAILABLE = False

_PIPELINE_CACHE = {}
_DIST_FIXED_CKPT_DIR = ""
_DIST_FIXED_WAV2VEC_DIR = ""
_DIST_GLOO_GROUP = None
_RTC_VIDEO_QUEUE = queue.Queue(maxsize=256) if _RTC_AVAILABLE else None
_RTC_AUDIO_QUEUE = queue.Queue(maxsize=4096) if _RTC_AVAILABLE else None
_RTC_ENABLED = False


def _rtc_output_frame_size(sample_rate):
    return sample_rate // 50


def _clear_rtc_queues():
    if not _RTC_AVAILABLE or not _RTC_ENABLED:
        return
    for q in (_RTC_VIDEO_QUEUE, _RTC_AUDIO_QUEUE):
        while True:
            try:
                q.get_nowait()
            except queue.Empty:
                break


def _enqueue_rtc_video(frame):
    if not _RTC_AVAILABLE or not _RTC_ENABLED:
        return
    try:
        _RTC_VIDEO_QUEUE.put_nowait(frame)
    except queue.Full:
        pass


def _enqueue_rtc_audio(audio_array, sample_rate):
    if not _RTC_AVAILABLE or not _RTC_ENABLED:
        return
    frame_size = _rtc_output_frame_size(sample_rate)
    audio_array = np.asarray(audio_array, dtype=np.float32)
    total = len(audio_array)
    idx = 0
    while idx + frame_size <= total:
        chunk = audio_array[idx : idx + frame_size]
        try:
            _RTC_AUDIO_QUEUE.put_nowait(chunk)
        except queue.Full:
            break
        idx += frame_size


if _RTC_AVAILABLE:
    class FlashTalkRTCHandler(AsyncAudioVideoStreamHandler):
        def __init__(self, output_sample_rate, fps):
            super().__init__(
                expected_layout="mono",
                output_sample_rate=output_sample_rate,
                input_sample_rate=output_sample_rate,
                fps=fps,
            )
            self.video_queue = _RTC_VIDEO_QUEUE
            self.audio_queue = _RTC_AUDIO_QUEUE

        def copy(self, **kwargs):
            return FlashTalkRTCHandler(
                output_sample_rate=self.output_sample_rate,
                fps=self.fps,
            )

        async def video_receive(self, frame: np.ndarray):
            return

        async def video_emit(self) -> VideoEmitType:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, self.video_queue.get)

        async def receive(self, frame: tuple[int, np.ndarray]) -> None:
            return

        async def emit(self) -> AudioEmitType:
            if not self.args_set.is_set():
                await self.wait_for_args()
            loop = asyncio.get_running_loop()
            array = await loop.run_in_executor(None, self.audio_queue.get)
            return (self.output_sample_rate, array)

        def shutdown(self) -> None:
            self.connection = None
            self.args_set.clear()


def _get_dist_info():
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    return world_size, rank


def _is_dist_enabled():
    world_size, _ = _get_dist_info()
    return world_size > 1 and dist.is_available()


def _get_object_group():
    if not dist.is_initialized():
        return None
    backend = dist.get_backend()
    if backend == "gloo":
        return dist.group.WORLD
    global _DIST_GLOO_GROUP
    if _DIST_GLOO_GROUP is None:
        try:
            _DIST_GLOO_GROUP = dist.new_group(backend="gloo")
        except Exception as exc:
            logger.warning("Failed to create gloo group, falling back to WORLD: {}", exc)
            _DIST_GLOO_GROUP = dist.group.WORLD
    return _DIST_GLOO_GROUP


def _init_object_group():
    if _is_dist_enabled() and dist.is_initialized():
        _get_object_group()


def _get_pipeline_cached(ckpt_dir, wav2vec_dir):
    cache_key = (ckpt_dir, wav2vec_dir)
    if cache_key in _PIPELINE_CACHE:
        return _PIPELINE_CACHE[cache_key]
    world_size, _ = _get_dist_info()
    pipeline = get_pipeline(world_size=world_size, ckpt_dir=ckpt_dir, wav2vec_dir=wav2vec_dir)
    _PIPELINE_CACHE[cache_key] = pipeline
    return pipeline


def _iter_audio_embeddings(pipeline, audio_array, audio_encode_mode):
    sample_rate = infer_params["sample_rate"]
    tgt_fps = infer_params["tgt_fps"]
    frame_num = infer_params["frame_num"]
    motion_frames_num = infer_params["motion_frames_num"]
    slice_len = frame_num - motion_frames_num
    audio_slice_len = slice_len * sample_rate // tgt_fps

    if audio_encode_mode == "once":
        audio_embedding_all = get_audio_embedding(pipeline, audio_array)
        total_chunks = (audio_embedding_all.shape[1] - frame_num) // slice_len
        for i in range(total_chunks):
            audio_start = i * audio_slice_len
            audio_end = audio_start + audio_slice_len
            audio_chunk = audio_array[audio_start:audio_end]
            yield audio_embedding_all[
                :,
                i * slice_len : i * slice_len + frame_num,
            ].contiguous(), audio_chunk
        return

    cached_audio_duration = infer_params["cached_audio_duration"]
    cached_audio_length_sum = sample_rate * cached_audio_duration
    audio_end_idx = cached_audio_duration * tgt_fps
    audio_start_idx = audio_end_idx - frame_num

    audio_dq = deque([0.0] * cached_audio_length_sum, maxlen=cached_audio_length_sum)
    audio_slices = audio_array[: (len(audio_array) // audio_slice_len) * audio_slice_len]
    audio_slices = audio_slices.reshape(-1, audio_slice_len)

    for audio_slice in audio_slices:
        audio_dq.extend(audio_slice.tolist())
        audio_window = np.array(audio_dq)
        yield get_audio_embedding(pipeline, audio_window, audio_start_idx, audio_end_idx), audio_slice


def _merge_audio_video(video_path, audio_path, output_path):
    cmd = [
        "ffmpeg",
        "-i",
        video_path,
        "-i",
        audio_path,
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        output_path,
        "-y",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.warning("ffmpeg merge failed: {}", result.stderr.strip())
        return None
    return output_path


def _broadcast_request(payload):
    if not _is_dist_enabled() or not dist.is_initialized():
        return
    group = _get_object_group()
    logger.info("Rank0 broadcasting request: {}", payload.get("cmd"))
    obj_list = [payload]
    dist.broadcast_object_list(obj_list, src=0, group=group)


def _stream_job(
    ckpt_dir,
    wav2vec_dir,
    input_prompt,
    cond_image_path,
    audio_path,
    audio_encode_mode,
    base_seed,
    write_video,
    stream_output,
):
    base_seed = base_seed if base_seed >= 0 else 9999
    pipeline = _get_pipeline_cached(ckpt_dir, wav2vec_dir)
    get_base_data(
        pipeline,
        input_prompt=input_prompt,
        cond_image=cond_image_path,
        base_seed=base_seed,
    )

    sample_rate = infer_params["sample_rate"]
    tgt_fps = infer_params["tgt_fps"]
    audio_array, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
    if stream_output:
        _clear_rtc_queues()

    final_output_path = None
    temp_video_path = None
    writer = None
    if write_video:
        output_dir = "sample_results"
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H:%M:%S-%f")[:-3]
        final_output_path = os.path.join(output_dir, f"res_{timestamp}.mp4")

        temp_dir = tempfile.mkdtemp(prefix="flashtalk_stream_")
        temp_video_path = os.path.join(temp_dir, "temp_video.mp4")

        writer = imageio.get_writer(
            temp_video_path,
            format="mp4",
            mode="I",
            fps=tgt_fps,
            codec="h264",
            ffmpeg_params=["-bf", "0"],
        )

    last_frame = None
    try:
        for chunk_idx, (audio_embedding, audio_chunk) in enumerate(
            _iter_audio_embeddings(pipeline, audio_array, audio_encode_mode)
        ):
            if stream_output and audio_chunk is not None:
                _enqueue_rtc_audio(audio_chunk, sample_rate)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.time()

            with torch.inference_mode():
                video = run_pipeline(pipeline, audio_embedding)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.time()
            logger.info(
                "Generate video chunk-{} done, cost time: {:.2f}s",
                chunk_idx,
                end_time - start_time,
            )

            if write_video or stream_output:
                frames = video.detach().cpu().numpy().astype(np.uint8)
                for frame in frames:
                    frame = np.ascontiguousarray(frame)
                    if write_video:
                        writer.append_data(frame)
                    last_frame = frame
                if stream_output:
                    logger.info("Yield live frame for chunk {}", chunk_idx)
                    if last_frame is not None:
                        _enqueue_rtc_video(last_frame)
                        yield Image.fromarray(last_frame), None
    finally:
        if writer is not None:
            writer.close()

    if not write_video:
        if _is_dist_enabled() and dist.is_initialized():
            dist.barrier()
        return

    if last_frame is None:
        raise gr.Error("No frames generated. Please check your inputs.")

    merged_path = _merge_audio_video(temp_video_path, audio_path, final_output_path)
    final_video_path = merged_path or temp_video_path
    if _is_dist_enabled() and dist.is_initialized() and not stream_output:
        dist.barrier()
    if last_frame is not None:
        _enqueue_rtc_video(last_frame)
        yield Image.fromarray(last_frame), final_video_path


def stream_generate(
    ckpt_dir,
    wav2vec_dir,
    input_prompt,
    cond_image_path,
    audio_path,
    audio_encode_mode,
    base_seed,
):
    if not ckpt_dir or not wav2vec_dir:
        raise gr.Error("Please set both ckpt_dir and wav2vec_dir.")
    if not cond_image_path or not os.path.exists(cond_image_path):
        raise gr.Error("Please upload a valid condition image.")
    if not audio_path or not os.path.exists(audio_path):
        raise gr.Error("Please upload a valid audio file.")

    if _is_dist_enabled():
        if _DIST_FIXED_CKPT_DIR:
            ckpt_dir = _DIST_FIXED_CKPT_DIR
        if _DIST_FIXED_WAV2VEC_DIR:
            wav2vec_dir = _DIST_FIXED_WAV2VEC_DIR
        _broadcast_request(
            {
                "cmd": "start",
                "ckpt_dir": ckpt_dir,
                "wav2vec_dir": wav2vec_dir,
                "input_prompt": input_prompt,
                "cond_image_path": cond_image_path,
                "audio_path": audio_path,
                "audio_encode_mode": audio_encode_mode,
                "base_seed": base_seed,
            }
        )

    yield from _stream_job(
        ckpt_dir=ckpt_dir,
        wav2vec_dir=wav2vec_dir,
        input_prompt=input_prompt,
        cond_image_path=cond_image_path,
        audio_path=audio_path,
        audio_encode_mode=audio_encode_mode,
        base_seed=base_seed,
        write_video=True,
        stream_output=True,
    )


def _worker_loop():
    logger.info("Multi-GPU worker ready, waiting for requests.")
    group = _get_object_group()
    while True:
        payload_list = [None]
        logger.info("Worker waiting for broadcast from rank0.")
        dist.broadcast_object_list(payload_list, src=0, group=group)
        payload = payload_list[0]
        logger.info("Worker received payload: {}", payload.get("cmd") if payload else None)
        if payload is None or payload.get("cmd") == "shutdown":
            return
        for _ in _stream_job(
            ckpt_dir=payload["ckpt_dir"],
            wav2vec_dir=payload["wav2vec_dir"],
            input_prompt=payload["input_prompt"],
            cond_image_path=payload["cond_image_path"],
            audio_path=payload["audio_path"],
            audio_encode_mode=payload["audio_encode_mode"],
            base_seed=payload["base_seed"],
            write_video=False,
            stream_output=False,
        ):
            pass


def build_demo(default_ckpt_dir, default_wav2vec_dir, multi_gpu):
    with gr.Blocks(title="SoulX-FlashTalk Streaming Demo") as demo:
        gr.Markdown(
            "# Streaming Avatar Demo\n"
            "Upload a condition image and an audio clip to stream frames as they are generated."
        )
        if multi_gpu:
            gr.Markdown("Multi-GPU mode enabled. Use CLI args to set model paths.")
        with gr.Row():
            with gr.Column(scale=1):
                ckpt_dir = gr.Textbox(
                    label="Checkpoint dir",
                    value=default_ckpt_dir,
                    placeholder="models/SoulX-FlashTalk-14B",
                    interactive=not multi_gpu,
                )
                wav2vec_dir = gr.Textbox(
                    label="Wav2Vec dir",
                    value=default_wav2vec_dir,
                    placeholder="models/chinese-wav2vec2-base",
                    interactive=not multi_gpu,
                )
                input_prompt = gr.Textbox(
                    label="Prompt",
                    value=(
                        "A person is talking. Only the foreground characters are moving, "
                        "the background remains static."
                    ),
                )
                base_seed = gr.Number(label="Base seed", value=9999, precision=0)
                cond_image = gr.Image(
                    label="Condition image",
                    type="filepath",
                    value="examples/man.png",
                )
                audio = gr.Audio(
                    label="Audio",
                    type="filepath",
                    value="examples/cantonese_16k.wav",
                )
                audio_encode_mode = gr.Dropdown(
                    label="Audio encode mode",
                    choices=["stream", "once"],
                    value="stream",
                )
                gr.Examples(
                    examples=[
                        ["examples/man.png", "examples/cantonese_16k.wav"],
                    ],
                    inputs=[cond_image, audio],
                    label="Examples",
                )
                run_btn = gr.Button("Generate (Stream)")

            with gr.Column(scale=1):
                stream_frame = gr.Image(
                    label="Live frame",
                    streaming=True,
                    show_label=True,
                    type="pil",
                    image_mode="RGB",
                )
                final_video = gr.Video(
                    label="Final video",
                    format="mp4",
                    show_label=True,
                )
                if _RTC_AVAILABLE:
                    rtc = WebRTC(
                        label="WebRTC Stream",
                        modality="audio-video",
                        mode="send-receive",
                        video_chat=True,
                        elem_id="rtc-output",
                    )
                    handler = FlashTalkRTCHandler(
                        output_sample_rate=infer_params["sample_rate"],
                        fps=infer_params["tgt_fps"],
                    )
                    rtc.stream(
                        handler,
                        inputs=[rtc],
                        outputs=[rtc],
                        time_limit=3600,
                        concurrency_limit=1,
                    )

        run_btn.click(
            stream_generate,
            inputs=[
                ckpt_dir,
                wav2vec_dir,
                input_prompt,
                cond_image,
                audio,
                audio_encode_mode,
                base_seed,
            ],
            outputs=[stream_frame, final_video],
        )

    try:
        demo.queue(concurrency_count=1, max_size=2)
    except TypeError:
        demo.queue(max_size=2)
    return demo


def _parse_args():
    parser = argparse.ArgumentParser(description="SoulX-FlashTalk Gradio streaming demo.")
    parser.add_argument("--ckpt_dir", type=str, default="", help="FlashTalk checkpoint dir.")
    parser.add_argument("--wav2vec_dir", type=str, default="", help="Wav2Vec checkpoint dir.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Gradio host.")
    parser.add_argument("--port", type=int, default=7860, help="Gradio port.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    world_size, rank = _get_dist_info()
    if world_size > 1 and (not args.ckpt_dir or not args.wav2vec_dir):
        raise SystemExit("Multi-GPU requires --ckpt_dir and --wav2vec_dir.")

    if world_size > 1:
        _DIST_FIXED_CKPT_DIR = args.ckpt_dir
        _DIST_FIXED_WAV2VEC_DIR = args.wav2vec_dir
        _get_pipeline_cached(args.ckpt_dir, args.wav2vec_dir)
        _init_object_group()

    if world_size > 1 and rank != 0:
        _worker_loop()
    else:
        if _RTC_AVAILABLE:
            _RTC_ENABLED = True
        app = build_demo(args.ckpt_dir, args.wav2vec_dir, world_size > 1)
        allowed_paths = [
            os.path.abspath("examples"),
            os.path.abspath("sample_results"),
        ]
        app.launch(server_name=args.host, server_port=args.port, allowed_paths=allowed_paths)
