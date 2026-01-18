import os
import queue
import sys
import threading
from collections import deque
from typing import Dict, Optional, cast, Tuple

import librosa
import numpy as np
import torch
import yaml
from loguru import logger
from pydantic import BaseModel, Field

from chat_engine.common.handler_base import HandlerBase, HandlerBaseInfo, HandlerDataInfo, HandlerDetail
from chat_engine.contexts.handler_context import HandlerContext
from chat_engine.contexts.session_context import SessionContext
from chat_engine.data_models.chat_data.chat_data_model import ChatData
from chat_engine.data_models.chat_data_type import ChatDataType
from chat_engine.data_models.chat_engine_config_data import ChatEngineConfigModel, HandlerBaseConfigModel
from chat_engine.data_models.runtime_data.data_bundle import DataBundleDefinition, DataBundleEntry, DataBundle, VariableSize
from engine_utils.directory_info import DirectoryInfo


def _ensure_flash_talk_on_path() -> str:
    project_dir = DirectoryInfo.get_project_dir()
    repo_root = os.path.abspath(os.path.join(project_dir, "..", "..", ".."))
    avatar_root = os.path.join(repo_root, "src", "avatar")
    if os.path.isdir(avatar_root) and avatar_root not in sys.path:
        sys.path.append(avatar_root)
    return avatar_root


_AVATAR_ROOT = _ensure_flash_talk_on_path()
_INFER_PARAMS_PATH = os.path.join(_AVATAR_ROOT, "flash_talk", "configs", "infer_params.yaml")

with open(_INFER_PARAMS_PATH, "r") as f:
    _INFER_PARAMS = yaml.safe_load(f)

from flash_talk.src.pipeline.flash_talk_pipeline import FlashTalkPipeline  # noqa: E402
from flash_talk.src.distributed.usp_device import get_device, get_parallel_degree  # noqa: E402
from flash_talk.infinite_talk.configs import multitalk_14B  # noqa: E402
from flash_talk.infinite_talk.utils.multitalk_utils import loudness_norm  # noqa: E402


class AvatarFlashTalkConfig(HandlerBaseConfigModel, BaseModel):
    ckpt_dir: str = Field(default="models/SoulX-FlashTalk-14B")
    wav2vec_dir: str = Field(default="models/chinese-wav2vec2-base")
    cond_image_path: str = Field(default="src/avatar/examples/man.png")
    input_prompt: str = Field(default="A person is talking.")
    audio_encode_mode: str = Field(default="stream")
    base_seed: int = Field(default=9999)


class AvatarFlashTalkContext(HandlerContext):
    def __init__(self, session_id: str):
        super().__init__(session_id)
        self.config: Optional[AvatarFlashTalkConfig] = None
        self.audio_queue: queue.Queue = queue.Queue()
        self.worker_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()

        self.current_speech_id: Optional[str] = None
        self.audio_buffer: np.ndarray = np.zeros((0,), dtype=np.float32)
        self.audio_deque: Optional[deque] = None
        self.base_ready = False


class HandlerAvatarFlashTalk(HandlerBase):
    def __init__(self) -> None:
        super().__init__()
        self.pipeline: Optional[FlashTalkPipeline] = None
        self.pipeline_lock = threading.Lock()
        self.output_data_definitions: Dict[ChatDataType, DataBundleDefinition] = {}

    def get_handler_info(self) -> HandlerBaseInfo:
        return HandlerBaseInfo(
            config_model=AvatarFlashTalkConfig,
            load_priority=-999,
        )

    def _get_pipeline(self, ckpt_dir: str, wav2vec_dir: str) -> FlashTalkPipeline:
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        config = multitalk_14B
        ulysses_degree, ring_degree = get_parallel_degree(world_size, config.num_heads)
        device = get_device(ulysses_degree, ring_degree)
        logger.info("FlashTalk device: {}", device)
        return FlashTalkPipeline(
            config=config,
            checkpoint_dir=ckpt_dir,
            wav2vec_dir=wav2vec_dir,
            device=device,
            use_usp=(world_size > 1),
        )

    def load(self, engine_config: ChatEngineConfigModel, handler_config: Optional[AvatarFlashTalkConfig] = None):
        if not isinstance(handler_config, AvatarFlashTalkConfig):
            handler_config = AvatarFlashTalkConfig()

        project_dir = DirectoryInfo.get_project_dir()
        repo_root = os.path.abspath(os.path.join(project_dir, "..", "..", ".."))

        if not os.path.isabs(handler_config.ckpt_dir):
            ckpt_dir = os.path.join(project_dir, handler_config.ckpt_dir)
            if not os.path.isdir(ckpt_dir):
                ckpt_dir = os.path.join(repo_root, handler_config.ckpt_dir)
        else:
            ckpt_dir = handler_config.ckpt_dir
        if not os.path.isabs(handler_config.wav2vec_dir):
            wav2vec_dir = os.path.join(project_dir, handler_config.wav2vec_dir)
            if not os.path.isdir(wav2vec_dir):
                wav2vec_dir = os.path.join(repo_root, handler_config.wav2vec_dir)
        else:
            wav2vec_dir = handler_config.wav2vec_dir

        if not os.path.isdir(ckpt_dir):
            raise FileNotFoundError(f"FlashTalk ckpt_dir not found: {ckpt_dir}")
        if not os.path.isdir(wav2vec_dir):
            raise FileNotFoundError(f"FlashTalk wav2vec_dir not found: {wav2vec_dir}")
        if not os.path.isabs(handler_config.cond_image_path):
            cond_image_path = os.path.join(project_dir, handler_config.cond_image_path)
            if not os.path.exists(cond_image_path):
                cond_image_path = os.path.join(repo_root, handler_config.cond_image_path)
        else:
            cond_image_path = handler_config.cond_image_path
        if not os.path.exists(cond_image_path):
            raise FileNotFoundError(f"FlashTalk cond_image_path not found: {cond_image_path}")

        handler_config.ckpt_dir = ckpt_dir
        handler_config.wav2vec_dir = wav2vec_dir
        handler_config.cond_image_path = cond_image_path

        video_output_definition = DataBundleDefinition()
        video_output_definition.add_entry(DataBundleEntry.create_framed_entry(
            "avatar_flash_video",
            [VariableSize(), VariableSize(), VariableSize(), 3],
            0,
            _INFER_PARAMS["tgt_fps"],
        ))
        video_output_definition.lockdown()
        self.output_data_definitions[ChatDataType.AVATAR_VIDEO] = video_output_definition

        self.pipeline = self._get_pipeline(ckpt_dir, wav2vec_dir)
        logger.info("HandlerAvatarFlashTalk loaded.")

    def create_context(self, session_context: SessionContext,
                       handler_config: Optional[AvatarFlashTalkConfig] = None) -> HandlerContext:
        if not isinstance(handler_config, AvatarFlashTalkConfig):
            handler_config = AvatarFlashTalkConfig()
        context = AvatarFlashTalkContext(session_context.session_info.session_id)
        context.config = handler_config
        return context

    def start_context(self, session_context: SessionContext, handler_context: HandlerContext):
        context = cast(AvatarFlashTalkContext, handler_context)
        context.worker_thread = threading.Thread(
            target=self._worker_loop,
            args=(context,),
            daemon=True,
        )
        context.worker_thread.start()

    def get_handler_detail(self, session_context: SessionContext,
                           context: HandlerContext) -> HandlerDetail:
        inputs = {
            ChatDataType.AVATAR_AUDIO: HandlerDataInfo(
                type=ChatDataType.AVATAR_AUDIO,
            )
        }
        outputs = {
            ChatDataType.AVATAR_VIDEO: HandlerDataInfo(
                type=ChatDataType.AVATAR_VIDEO,
                definition=self.output_data_definitions[ChatDataType.AVATAR_VIDEO],
            ),
        }
        return HandlerDetail(inputs=inputs, outputs=outputs)

    def handle(self, context: HandlerContext, inputs: ChatData,
               output_definitions: Dict[ChatDataType, HandlerDataInfo]):
        if inputs.type != ChatDataType.AVATAR_AUDIO:
            return
        context = cast(AvatarFlashTalkContext, context)
        speech_id = inputs.data.get_meta("speech_id") or context.session_id
        speech_end = inputs.data.get_meta("avatar_speech_end", False)
        audio_entry = inputs.data.get_main_definition_entry()
        audio_array = inputs.data.get_main_data()
        if audio_array is None:
            return
        audio_array = np.asarray(audio_array, dtype=np.float32).squeeze()
        context.audio_queue.put_nowait((speech_id, audio_entry.sample_rate, audio_array, speech_end))

    def _prepare_base(self, context: AvatarFlashTalkContext):
        if self.pipeline is None:
            raise RuntimeError("FlashTalk pipeline not initialized.")
        self.pipeline.prepare_params(
            input_prompt=context.config.input_prompt,
            cond_image=context.config.cond_image_path,
            target_size=(
                _INFER_PARAMS["height"],
                _INFER_PARAMS["width"],
            ),
            frame_num=_INFER_PARAMS["frame_num"],
            motion_frames_num=_INFER_PARAMS["motion_frames_num"],
            sampling_steps=_INFER_PARAMS["sample_steps"],
            seed=context.config.base_seed,
            shift=_INFER_PARAMS["sample_shift"],
            color_correction_strength=_INFER_PARAMS["color_correction_strength"],
        )

    def _get_audio_embedding(self, audio_array: np.ndarray,
                             audio_start_idx: int = -1,
                             audio_end_idx: int = -1) -> torch.Tensor:
        audio_array = loudness_norm(audio_array, _INFER_PARAMS["sample_rate"])
        audio_embedding = self.pipeline.preprocess_audio(
            audio_array,
            sr=_INFER_PARAMS["sample_rate"],
            fps=_INFER_PARAMS["tgt_fps"],
        )
        if audio_start_idx == -1 or audio_end_idx == -1:
            audio_start_idx = 0
            audio_end_idx = audio_embedding.shape[0]
        indices = (torch.arange(2 * 2 + 1) - 2) * 1
        center_indices = torch.arange(audio_start_idx, audio_end_idx, 1).unsqueeze(1) + indices.unsqueeze(0)
        center_indices = torch.clamp(center_indices, min=0, max=audio_end_idx - 1)
        return audio_embedding[center_indices][None, ...].contiguous()

    def _run_pipeline(self, audio_embedding: torch.Tensor) -> np.ndarray:
        audio_embedding = audio_embedding.to(self.pipeline.device)
        sample = self.pipeline.generate(audio_embedding)
        sample_frames = (((sample + 1) / 2).permute(1, 2, 3, 0).clip(0, 1) * 255).contiguous()
        return sample_frames.detach().cpu().numpy().astype(np.uint8)

    def _iter_embeddings_once(self, audio_array: np.ndarray):
        sample_rate = _INFER_PARAMS["sample_rate"]
        tgt_fps = _INFER_PARAMS["tgt_fps"]
        frame_num = _INFER_PARAMS["frame_num"]
        motion_frames_num = _INFER_PARAMS["motion_frames_num"]
        slice_len = frame_num - motion_frames_num
        audio_slice_len = slice_len * sample_rate // tgt_fps

        audio_embedding_all = self._get_audio_embedding(audio_array)
        total_chunks = (audio_embedding_all.shape[1] - frame_num) // slice_len
        for i in range(total_chunks):
            audio_start = i * audio_slice_len
            audio_end = audio_start + audio_slice_len
            audio_chunk = audio_array[audio_start:audio_end]
            yield audio_embedding_all[
                :,
                i * slice_len : i * slice_len + frame_num,
            ].contiguous(), audio_chunk

    def _iter_embeddings_stream(self, context: AvatarFlashTalkContext, audio_array: np.ndarray):
        sample_rate = _INFER_PARAMS["sample_rate"]
        tgt_fps = _INFER_PARAMS["tgt_fps"]
        frame_num = _INFER_PARAMS["frame_num"]
        motion_frames_num = _INFER_PARAMS["motion_frames_num"]
        slice_len = frame_num - motion_frames_num
        audio_slice_len = slice_len * sample_rate // tgt_fps

        if context.audio_deque is None:
            cached_audio_duration = _INFER_PARAMS["cached_audio_duration"]
            cached_audio_length_sum = sample_rate * cached_audio_duration
            context.audio_deque = deque([0.0] * cached_audio_length_sum, maxlen=cached_audio_length_sum)

        context.audio_buffer = np.concatenate([context.audio_buffer, audio_array], axis=0)
        audio_end_idx = _INFER_PARAMS["cached_audio_duration"] * tgt_fps
        audio_start_idx = audio_end_idx - frame_num

        while context.audio_buffer.shape[0] >= audio_slice_len:
            audio_slice = context.audio_buffer[:audio_slice_len]
            context.audio_buffer = context.audio_buffer[audio_slice_len:]
            context.audio_deque.extend(audio_slice.tolist())
            audio_window = np.array(context.audio_deque)
            audio_embedding = self._get_audio_embedding(audio_window, audio_start_idx, audio_end_idx)
            yield audio_embedding, audio_slice

    def _flush_stream_remainder(self, context: AvatarFlashTalkContext):
        sample_rate = _INFER_PARAMS["sample_rate"]
        tgt_fps = _INFER_PARAMS["tgt_fps"]
        frame_num = _INFER_PARAMS["frame_num"]
        motion_frames_num = _INFER_PARAMS["motion_frames_num"]
        slice_len = frame_num - motion_frames_num
        audio_slice_len = slice_len * sample_rate // tgt_fps

        if context.audio_buffer.shape[0] == 0:
            return None
        remainder = context.audio_buffer
        if remainder.shape[0] < audio_slice_len:
            pad_len = audio_slice_len - remainder.shape[0]
            remainder = np.concatenate([remainder, np.zeros((pad_len,), dtype=np.float32)], axis=0)
        context.audio_buffer = np.zeros((0,), dtype=np.float32)
        if context.audio_deque is None:
            cached_audio_duration = _INFER_PARAMS["cached_audio_duration"]
            cached_audio_length_sum = sample_rate * cached_audio_duration
            context.audio_deque = deque([0.0] * cached_audio_length_sum, maxlen=cached_audio_length_sum)
        context.audio_deque.extend(remainder.tolist())
        audio_end_idx = _INFER_PARAMS["cached_audio_duration"] * tgt_fps
        audio_start_idx = audio_end_idx - frame_num
        audio_window = np.array(context.audio_deque)
        audio_embedding = self._get_audio_embedding(audio_window, audio_start_idx, audio_end_idx)
        return audio_embedding, remainder

    def _send_frames(self, context: AvatarFlashTalkContext, frames: np.ndarray):
        definition = self.output_data_definitions[ChatDataType.AVATAR_VIDEO]
        for frame in frames:
            data_bundle = DataBundle(definition)
            data_bundle.set_main_data(np.ascontiguousarray(frame)[np.newaxis, ...])
            context.submit_data(data_bundle)

    def _worker_loop(self, context: AvatarFlashTalkContext):
        while not context.stop_event.is_set():
            try:
                speech_id, sample_rate, audio_array, speech_end = context.audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if self.pipeline is None:
                logger.error("FlashTalk pipeline is not ready.")
                continue

            if context.current_speech_id != speech_id:
                context.current_speech_id = speech_id
                context.audio_buffer = np.zeros((0,), dtype=np.float32)
                context.audio_deque = None
                context.base_ready = False

            if sample_rate != _INFER_PARAMS["sample_rate"]:
                audio_array = librosa.resample(
                    audio_array,
                    orig_sr=sample_rate,
                    target_sr=_INFER_PARAMS["sample_rate"],
                )

            if not context.base_ready:
                with self.pipeline_lock:
                    self._prepare_base(context)
                context.base_ready = True

            if context.config.audio_encode_mode == "once":
                context.audio_buffer = np.concatenate([context.audio_buffer, audio_array], axis=0)
                if not speech_end:
                    continue
                audio_full = context.audio_buffer
                context.audio_buffer = np.zeros((0,), dtype=np.float32)
                with self.pipeline_lock:
                    for audio_embedding, _ in self._iter_embeddings_once(audio_full):
                        frames = self._run_pipeline(audio_embedding)
                        self._send_frames(context, frames)
            else:
                with self.pipeline_lock:
                    for audio_embedding, _ in self._iter_embeddings_stream(context, audio_array):
                        frames = self._run_pipeline(audio_embedding)
                        self._send_frames(context, frames)
                    if speech_end:
                        flush = self._flush_stream_remainder(context)
                        if flush is not None:
                            audio_embedding, _ = flush
                            frames = self._run_pipeline(audio_embedding)
                            self._send_frames(context, frames)

            if speech_end:
                context.current_speech_id = None
                context.base_ready = False

    def destroy_context(self, context: HandlerContext):
        context = cast(AvatarFlashTalkContext, context)
        context.stop_event.set()
        if context.worker_thread is not None:
            context.worker_thread.join(timeout=5)
            context.worker_thread = None
