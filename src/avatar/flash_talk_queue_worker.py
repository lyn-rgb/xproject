# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
import json
import os
import time
from collections import deque

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

_PIPELINE_CACHE = {}
_DIST_GLOO_GROUP = None


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


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _write_frame(frame, frame_dir, frame_idx):
    frame_path = os.path.join(frame_dir, f"frame_{frame_idx:06d}.jpg")
    Image.fromarray(frame).save(frame_path, format="JPEG", quality=90)


def _run_generation(payload, output_dir, rank):
    sample_rate = infer_params["sample_rate"]
    tgt_fps = infer_params["tgt_fps"]
    pipeline = _get_pipeline_cached(payload["ckpt_dir"], payload["wav2vec_dir"])
    get_base_data(
        pipeline,
        input_prompt=payload["input_prompt"],
        cond_image=payload["cond_image_path"],
        base_seed=payload["base_seed"],
    )

    audio_array, _ = librosa.load(payload["audio_path"], sr=sample_rate, mono=True)
    frame_dir = os.path.join(output_dir, "frames")
    if rank == 0:
        _ensure_dir(frame_dir)

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
            "Queue video chunk-{} done, cost time: {:.2f}s",
            chunk_idx,
            end_time - start_time,
        )
        if rank == 0:
            frames = video.detach().cpu().numpy().astype(np.uint8)
            for frame in frames:
                _write_frame(np.ascontiguousarray(frame), frame_dir, frame_idx)
                frame_idx += 1
    if _is_dist_enabled() and dist.is_initialized():
        dist.barrier()
    if rank == 0:
        done_path = os.path.join(output_dir, "DONE")
        with open(done_path, "w") as f:
            f.write("done\n")


def _broadcast_payload(payload):
    if not _is_dist_enabled() or not dist.is_initialized():
        return
    group = _get_object_group()
    obj_list = [payload]
    dist.broadcast_object_list(obj_list, src=0, group=group)


def _poll_jobs(queue_dir, rank):
    jobs_dir = os.path.join(queue_dir, "jobs")
    results_dir = os.path.join(queue_dir, "results")
    _ensure_dir(jobs_dir)
    _ensure_dir(results_dir)

    while True:
        if rank == 0:
            job_files = sorted(
                f for f in os.listdir(jobs_dir) if f.endswith(".json")
            )
            payload = None
            if job_files:
                job_file = os.path.join(jobs_dir, job_files[0])
                with open(job_file, "r") as f:
                    payload = json.load(f)
                os.remove(job_file)
        else:
            payload = None

        if _is_dist_enabled() and dist.is_initialized():
            payload_list = [payload]
            group = _get_object_group()
            dist.broadcast_object_list(payload_list, src=0, group=group)
            payload = payload_list[0]

        if payload is None:
            time.sleep(0.2)
            continue
        if payload.get("cmd") == "shutdown":
            return

        output_dir = payload.get("output_dir")
        if not output_dir:
            logger.error("Job payload missing output_dir")
            continue

        _run_generation(payload, output_dir, rank)


def _parse_args():
    parser = argparse.ArgumentParser(description="FlashTalk file queue worker.")
    parser.add_argument("--queue_dir", type=str, default="tmp/flashtalk_queue")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    _, rank = _get_dist_info()
    if _is_dist_enabled() and dist.is_initialized():
        _get_object_group()
    _poll_jobs(args.queue_dir, rank)
