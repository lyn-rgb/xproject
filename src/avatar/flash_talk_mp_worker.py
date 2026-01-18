# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc
import os
import time
from collections import deque
from typing import Dict, Optional

import librosa
import numpy as np
import torch
import torch.distributed as dist
from loguru import logger

from flash_talk.inference import (
    get_pipeline,
    get_base_data,
    get_audio_embedding,
    run_pipeline,
    infer_params,
)


def _get_pipeline_cached(cache: Dict, world_size: int, ckpt_dir: str, wav2vec_dir: str):
    cache_key = (ckpt_dir, wav2vec_dir)
    if cache_key in cache:
        return cache[cache_key]
    pipeline = get_pipeline(world_size=world_size, ckpt_dir=ckpt_dir, wav2vec_dir=wav2vec_dir)
    cache[cache_key] = pipeline
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
            yield audio_embedding_all[
                :,
                i * slice_len : i * slice_len + frame_num,
            ].contiguous(), audio_array[audio_start:audio_end]
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


def _broadcast_payload(payload, rank: int):
    payload_list = [payload]
    dist.broadcast_object_list(payload_list, src=0)
    return payload_list[0]


def _run_generation(payload, pipeline, rank, result_queue):
    sample_rate = infer_params["sample_rate"]
    tgt_fps = infer_params["tgt_fps"]

    get_base_data(
        pipeline,
        input_prompt=payload["input_prompt"],
        cond_image=payload["cond_image_path"],
        base_seed=payload["base_seed"],
    )

    audio_array = payload["audio_array"]
    input_sr = payload["sample_rate"]
    if input_sr != sample_rate:
        audio_array = librosa.resample(audio_array, orig_sr=input_sr, target_sr=sample_rate)

    frame_idx = 0
    for chunk_idx, (audio_embedding, _) in enumerate(
        _iter_audio_embeddings(pipeline, audio_array, payload["audio_encode_mode"])
    ):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.time()
        with torch.inference_mode():
            video = run_pipeline(pipeline, audio_embedding)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.time()
        logger.info(
            "MP video chunk-{} done, cost time: {:.2f}s",
            chunk_idx,
            end_time - start_time,
        )
        if rank == 0:
            frames = video.detach().cpu().numpy().astype(np.uint8)
            for frame in frames:
                result_queue.put(
                    {
                        "job_id": payload["job_id"],
                        "type": "frame",
                        "frame": frame,
                        "index": frame_idx,
                        "fps": tgt_fps,
                    }
                )
                frame_idx += 1
    dist.barrier()
    if rank == 0:
        result_queue.put({"job_id": payload["job_id"], "type": "done"})


def worker_loop(
    rank: int,
    world_size: int,
    init_method: str,
    job_queue,
    result_queue,
    ckpt_dir: str,
    wav2vec_dir: str,
    cuda_visible: str,
):
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible
    torch.cuda.set_device(rank)
    if not dist.is_initialized():
        if init_method.startswith("tcp://"):
            _, addr = init_method.split("://", 1)
            host, port = addr.split(":")
            os.environ.setdefault("MASTER_ADDR", host)
            os.environ.setdefault("MASTER_PORT", port)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank)

    pipeline_cache: Dict = {}
    pipeline = _get_pipeline_cached(pipeline_cache, world_size, ckpt_dir, wav2vec_dir)

    while True:
        payload = None
        if rank == 0:
            payload = job_queue.get()
        payload = _broadcast_payload(payload, rank)
        if payload is None or payload.get("cmd") == "shutdown":
            return
        _run_generation(payload, pipeline, rank, result_queue)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()
