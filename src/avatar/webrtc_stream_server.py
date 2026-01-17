# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
import asyncio
import os
import queue
import sys
import threading
import time
from collections import deque

import gradio as gr
import librosa
import numpy as np
import torch
from fastapi import HTTPException, Request
from loguru import logger

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

from fastrtc import (
    AsyncAudioVideoStreamHandler,
    WebRTC,
    VideoEmitType,
    AudioEmitType,
)

_VIDEO_QUEUE = queue.Queue(maxsize=512)
_AUDIO_QUEUE = queue.Queue(maxsize=8192)
_GEN_LOCK = threading.Lock()
_GEN_THREAD: threading.Thread | None = None


def _rtc_output_frame_size(sample_rate):
    return sample_rate // 50


def _clear_queues():
    for q in (_VIDEO_QUEUE, _AUDIO_QUEUE):
        while True:
            try:
                q.get_nowait()
            except queue.Empty:
                break


def _enqueue_video(frame):
    try:
        _VIDEO_QUEUE.put_nowait(frame)
    except queue.Full:
        try:
            _VIDEO_QUEUE.get_nowait()
        except queue.Empty:
            return
        try:
            _VIDEO_QUEUE.put_nowait(frame)
        except queue.Full:
            return


def _enqueue_audio(audio_array, sample_rate):
    frame_size = _rtc_output_frame_size(sample_rate)
    audio_array = np.asarray(audio_array, dtype=np.float32)
    total = len(audio_array)
    idx = 0
    while idx + frame_size <= total:
        chunk = audio_array[idx : idx + frame_size]
        try:
            _AUDIO_QUEUE.put_nowait(chunk)
        except queue.Full:
            return
        idx += frame_size


class FlashTalkRTCHandler(AsyncAudioVideoStreamHandler):
    def __init__(self, output_sample_rate, fps):
        super().__init__(
            expected_layout="mono",
            output_sample_rate=output_sample_rate,
            input_sample_rate=output_sample_rate,
            fps=fps,
        )

    def copy(self, **kwargs):
        return FlashTalkRTCHandler(
            output_sample_rate=self.output_sample_rate,
            fps=self.fps,
        )

    async def video_receive(self, frame: np.ndarray):
        return

    async def video_emit(self) -> VideoEmitType:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _VIDEO_QUEUE.get)

    async def receive(self, frame: tuple[int, np.ndarray]) -> None:
        return

    async def emit(self) -> AudioEmitType:
        if not self.args_set.is_set():
            await self.wait_for_args()
        loop = asyncio.get_running_loop()
        array = await loop.run_in_executor(None, _AUDIO_QUEUE.get)
        return (self.output_sample_rate, array)

    def shutdown(self) -> None:
        self.connection = None
        self.args_set.clear()


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


def _run_generation(payload):
    with _GEN_LOCK:
        _clear_queues()
        sample_rate = infer_params["sample_rate"]
        tgt_fps = infer_params["tgt_fps"]
        world_size = int(os.environ.get("WORLD_SIZE", 1))

        pipeline = get_pipeline(
            world_size=world_size,
            ckpt_dir=payload["ckpt_dir"],
            wav2vec_dir=payload["wav2vec_dir"],
        )
        get_base_data(
            pipeline,
            input_prompt=payload["input_prompt"],
            cond_image=payload["cond_image_path"],
            base_seed=payload["base_seed"],
        )

        audio_array, _ = librosa.load(payload["audio_path"], sr=sample_rate, mono=True)
        for chunk_idx, (audio_embedding, audio_chunk) in enumerate(
            _iter_audio_embeddings(pipeline, audio_array, payload["audio_encode_mode"])
        ):
            _enqueue_audio(audio_chunk, sample_rate)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.time()
            with torch.inference_mode():
                video = run_pipeline(pipeline, audio_embedding)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.time()
            logger.info(
                "RTC video chunk-{} done, cost time: {:.2f}s",
                chunk_idx,
                end_time - start_time,
            )
            frames = video.detach().cpu().numpy().astype(np.uint8)
            for frame in frames:
                _enqueue_video(np.ascontiguousarray(frame))
            time.sleep(1.0 / max(tgt_fps, 1))


def _start_generation(payload):
    global _GEN_THREAD
    if _GEN_THREAD is not None and _GEN_THREAD.is_alive():
        return False
    _GEN_THREAD = threading.Thread(target=_run_generation, args=(payload,), daemon=True)
    _GEN_THREAD.start()
    return True


def build_app(default_ckpt_dir, default_wav2vec_dir):
    with gr.Blocks(title="FlashTalk WebRTC Stream") as demo:
        gr.Markdown("WebRTC streaming server (video + audio). Trigger generation via POST /start.")
        webrtc = WebRTC(
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
        webrtc.stream(
            handler,
            inputs=[webrtc],
            outputs=[webrtc],
            time_limit=3600,
            concurrency_limit=1,
        )

    @demo.app.post("/start")
    async def start_stream(request: Request):
        payload = await request.json()
        payload.setdefault("ckpt_dir", default_ckpt_dir)
        payload.setdefault("wav2vec_dir", default_wav2vec_dir)
        payload.setdefault("input_prompt", "A person is talking.")
        payload.setdefault("audio_encode_mode", "stream")
        payload.setdefault("base_seed", 9999)

        if not payload.get("ckpt_dir") or not payload.get("wav2vec_dir"):
            raise HTTPException(status_code=400, detail="ckpt_dir and wav2vec_dir required")
        if not payload.get("cond_image_path") or not os.path.exists(payload["cond_image_path"]):
            raise HTTPException(status_code=400, detail="cond_image_path invalid")
        if not payload.get("audio_path") or not os.path.exists(payload["audio_path"]):
            raise HTTPException(status_code=400, detail="audio_path invalid")

        if not _start_generation(payload):
            raise HTTPException(status_code=409, detail="generation already running")
        return {"status": "started"}

    return demo


def _parse_args():
    parser = argparse.ArgumentParser(description="FlashTalk WebRTC streaming server.")
    parser.add_argument("--ckpt_dir", type=str, default="", help="FlashTalk checkpoint dir.")
    parser.add_argument("--wav2vec_dir", type=str, default="", help="Wav2Vec checkpoint dir.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host.")
    parser.add_argument("--port", type=int, default=7861, help="Server port.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    app = build_app(args.ckpt_dir, args.wav2vec_dir)
    app.launch(server_name=args.host, server_port=args.port)
