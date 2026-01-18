import os
import sys
from abc import ABC
from typing import Optional, cast, Dict, List

import numpy as np
import torch
from loguru import logger
from PIL import Image
from pydantic import BaseModel, Field
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

from chat_engine.common.handler_base import HandlerBase, HandlerDetail, HandlerBaseInfo, HandlerDataInfo
from chat_engine.contexts.handler_context import HandlerContext
from chat_engine.contexts.session_context import SessionContext
from chat_engine.data_models.chat_data.chat_data_model import ChatData
from chat_engine.data_models.chat_data_type import ChatDataType
from chat_engine.data_models.chat_engine_config_data import ChatEngineConfigModel, HandlerBaseConfigModel
from chat_engine.data_models.runtime_data.data_bundle import DataBundle, DataBundleDefinition, DataBundleEntry
from engine_utils.directory_info import DirectoryInfo


def _ensure_thinker_utils_on_path() -> str:
    project_dir = DirectoryInfo.get_project_dir()
    repo_root = os.path.abspath(os.path.join(project_dir, "..", "..", ".."))
    utils_path = os.path.join(repo_root, "src", "thinker", "qwen-omni-utils", "src")
    if os.path.isdir(utils_path) and utils_path not in sys.path:
        sys.path.append(utils_path)
    return utils_path


_THINKER_UTILS_PATH = _ensure_thinker_utils_on_path()

try:
    from qwen_omni_utils import process_mm_info
except Exception as exc:
    raise ImportError(
        f"Failed to import qwen_omni_utils from {_THINKER_UTILS_PATH}: {exc}"
    ) from exc


class ThinkerConfig(HandlerBaseConfigModel, BaseModel):
    model_name: str = Field(default="Qwen2.5-Omni-7B")
    system_prompt: str = Field(
        default=(
            "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
            "capable of perceiving auditory and visual inputs, as well as generating text and speech."
        )
    )
    voice: str = Field(default="Chelsie")
    enable_video_input: bool = Field(default=True)
    use_audio_in_video: bool = Field(default=True)


class ThinkerContext(HandlerContext):
    def __init__(self, session_id: str):
        super().__init__(session_id)
        self.config: Optional[ThinkerConfig] = None
        self.audio_segments: List[np.ndarray] = []
        self.last_video_frame: Optional[ChatData] = None


class HandlerThinkerS2S(HandlerBase, ABC):
    def __init__(self):
        super().__init__()
        self.model: Optional[Qwen2_5OmniForConditionalGeneration] = None
        self.processor: Optional[Qwen2_5OmniProcessor] = None

    def get_handler_info(self) -> HandlerBaseInfo:
        return HandlerBaseInfo(
            config_model=ThinkerConfig,
        )

    def load(self, engine_config: ChatEngineConfigModel, handler_config: Optional[BaseModel] = None):
        model_name = "Qwen2.5-Omni-7B"
        if isinstance(handler_config, ThinkerConfig):
            model_name = handler_config.model_name
        model_path = model_name
        if not os.path.isabs(model_path):
            model_path = os.path.join(engine_config.model_root, model_name)
        if not os.path.isdir(model_path):
            raise FileNotFoundError(f"Thinker model path not found: {model_path}")

        device_map = "auto"
        logger.info(f"Loading Thinker model from {model_path}")
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_path,
            device_map=device_map,
            torch_dtype="auto",
        )
        self.processor = Qwen2_5OmniProcessor.from_pretrained(model_path)

    def create_context(self, session_context: SessionContext,
                       handler_config: Optional[BaseModel] = None) -> HandlerContext:
        if not isinstance(handler_config, ThinkerConfig):
            handler_config = ThinkerConfig()
        context = ThinkerContext(session_context.session_info.session_id)
        context.config = handler_config
        return context

    def start_context(self, session_context, handler_context):
        pass

    def get_handler_detail(self, session_context: SessionContext,
                           context: HandlerContext) -> HandlerDetail:
        audio_definition = DataBundleDefinition()
        audio_definition.add_entry(DataBundleEntry.create_audio_entry("avatar_audio", 1, 24000))
        text_definition = DataBundleDefinition()
        text_definition.add_entry(DataBundleEntry.create_text_entry("avatar_text"))
        inputs = {
            ChatDataType.HUMAN_AUDIO: HandlerDataInfo(
                type=ChatDataType.HUMAN_AUDIO,
            ),
            ChatDataType.CAMERA_VIDEO: HandlerDataInfo(
                type=ChatDataType.CAMERA_VIDEO,
            ),
        }
        outputs = {
            ChatDataType.AVATAR_AUDIO: HandlerDataInfo(
                type=ChatDataType.AVATAR_AUDIO,
                definition=audio_definition,
            ),
            ChatDataType.AVATAR_TEXT: HandlerDataInfo(
                type=ChatDataType.AVATAR_TEXT,
                definition=text_definition,
            ),
        }
        return HandlerDetail(
            inputs=inputs, outputs=outputs,
        )

    @staticmethod
    def _frame_to_image(frame_array: np.ndarray) -> Image.Image:
        frame_array = np.squeeze(frame_array)
        return Image.fromarray(frame_array[..., ::-1])

    def _build_messages(self, context: ThinkerContext, audio: np.ndarray) -> List[Dict]:
        system_message = {
            "role": "system",
            "content": [
                {"type": "text", "text": context.config.system_prompt},
            ],
        }
        user_content = [
            {"type": "audio", "audio": audio},
        ]
        if context.config.enable_video_input and context.last_video_frame is not None:
            frame_array = context.last_video_frame.data.get_main_data()
            if frame_array is not None:
                user_content.insert(0, {"type": "image", "image": self._frame_to_image(frame_array)})
        user_message = {
            "role": "user",
            "content": user_content,
        }
        return [system_message, user_message]

    def handle(self, context: HandlerContext, inputs: ChatData,
               output_definitions: Dict[ChatDataType, HandlerDataInfo]):
        context = cast(ThinkerContext, context)
        if inputs.type == ChatDataType.CAMERA_VIDEO:
            if context.config is not None and context.config.enable_video_input:
                context.last_video_frame = inputs
            return

        if inputs.type != ChatDataType.HUMAN_AUDIO:
            return

        audio = inputs.data.get_main_data()
        if audio is not None:
            context.audio_segments.append(audio.squeeze().astype(np.float32))

        speech_end = inputs.data.get_meta("human_speech_end", False)
        if not speech_end:
            return

        if not context.audio_segments:
            return

        full_audio = np.concatenate(context.audio_segments, axis=0)
        context.audio_segments.clear()
        speech_id = inputs.data.get_meta("speech_id") or context.session_id

        if self.model is None or self.processor is None:
            logger.error("Thinker model is not loaded.")
            return

        messages = self._build_messages(context, full_audio)
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(messages, use_audio_in_video=context.config.use_audio_in_video)
        model_inputs = self.processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=context.config.use_audio_in_video,
        )
        model_inputs = model_inputs.to(self.model.device).to(self.model.dtype)

        try:
            text_ids, audio_out = self.model.generate(
                **model_inputs,
                speaker=context.config.voice,
                use_audio_in_video=context.config.use_audio_in_video,
            )
        except Exception as exc:
            logger.error(f"Thinker inference failed: {exc}")
            return

        response_text = self.processor.batch_decode(
            text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        response_text = response_text[0].split("\n")[-1] if response_text else ""

        text_definition = output_definitions.get(ChatDataType.AVATAR_TEXT).definition
        if text_definition is not None:
            text_output = DataBundle(text_definition)
            text_output.set_main_data(response_text)
            text_output.add_meta("avatar_text_end", False)
            text_output.add_meta("speech_id", speech_id)
            context.submit_data(text_output)
            end_text_output = DataBundle(text_definition)
            end_text_output.set_main_data("")
            end_text_output.add_meta("avatar_text_end", True)
            end_text_output.add_meta("speech_id", speech_id)
            context.submit_data(end_text_output)

        if isinstance(audio_out, torch.Tensor):
            audio_out = audio_out.detach().cpu().numpy()
        audio_out = np.array(audio_out, dtype=np.float32).reshape(-1)

        audio_definition = output_definitions.get(ChatDataType.AVATAR_AUDIO).definition
        if audio_definition is not None:
            audio_output = DataBundle(audio_definition)
            audio_output.set_main_data(audio_out[np.newaxis, ...])
            audio_output.add_meta("avatar_speech_end", False)
            audio_output.add_meta("speech_id", speech_id)
            context.submit_data(audio_output)
            end_audio_output = DataBundle(audio_definition)
            end_audio_output.set_main_data(np.zeros(shape=(1, 50), dtype=np.float32))
            end_audio_output.add_meta("avatar_speech_end", True)
            end_audio_output.add_meta("speech_id", speech_id)
            context.submit_data(end_audio_output)

    def destroy_context(self, context: HandlerContext):
        pass
