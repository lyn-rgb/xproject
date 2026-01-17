from modeling_qwen2_5_omni_low_VRAM_mode import Qwen2_5OmniForConditionalGeneration

from transformers import Qwen2_5OmniProcessor
from transformers.utils.hub import cached_file

from gptqmodel import GPTQModel
from gptqmodel.models.base import BaseGPTQModel
from gptqmodel.models.auto import MODEL_MAP
from gptqmodel.models._const import CPU, SUPPORTED_MODELS
from huggingface_hub import snapshot_download

from qwen_omni_utils import process_mm_info
from typing import Any, Dict

import torch
import time
import soundfile as sf

model_path = "Qwen/Qwen2.5-Omni-7B-GPTQ-Int4"
model_path = snapshot_download(repo_id=model_path) # if you use local model file, delete this line

class Qwen25OmniThinkerGPTQ(BaseGPTQModel):
    loader = Qwen2_5OmniForConditionalGeneration
    base_modules = [
        "thinker.model.embed_tokens", 
        "thinker.model.norm", 
        "token2wav", 
        "thinker.audio_tower", 
        "thinker.model.rotary_emb",
        "thinker.visual", 
        "talker"
    ]
    pre_lm_head_norm_module = "thinker.model.norm"
    require_monkeypatch = False
    layers_node = "thinker.model.layers"
    layer_type = "Qwen2_5OmniDecoderLayer"
    layer_modules = [
        ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
        ["self_attn.o_proj"],
        ["mlp.up_proj", "mlp.gate_proj"],
        ["mlp.down_proj"],
    ]
   
    def pre_quantize_generate_hook_start(self):
        self.thinker.visual = move_to(self.thinker.visual, device=self.quantize_config.device)
        self.thinker.audio_tower = move_to(self.thinker.audio_tower, device=self.quantize_config.device)

    def pre_quantize_generate_hook_end(self):
        self.thinker.visual = move_to(self.thinker.visual, device=CPU)
        self.thinker.audio_tower = move_to(self.thinker.audio_tower, device=CPU)

    def preprocess_dataset(self, sample: Dict) -> Dict:
        return sample
    

MODEL_MAP["qwen2_5_omni"] = Qwen25OmniThinkerGPTQ
SUPPORTED_MODELS.extend(["qwen2_5_omni"])

@classmethod
def patched_from_config(cls, config, *args, **kwargs):
    kwargs.pop("trust_remote_code", None)

    
    model = cls._from_config(config, **kwargs)
    spk_path = cached_file(
        model_path,
        "spk_dict.pt",
        subfolder=kwargs.pop("subfolder", None),
        cache_dir=kwargs.pop("cache_dir", None),
        force_download=kwargs.pop("force_download", False),
        proxies=kwargs.pop("proxies", None),
        resume_download=kwargs.pop("resume_download", None),
        local_files_only=kwargs.pop("local_files_only", False),
        token=kwargs.pop("use_auth_token", None),
        revision=kwargs.pop("revision", None),
    )
    if spk_path is None:
        raise ValueError(f"Speaker dictionary not found at {spk_path}")
    
    model.load_speakers(spk_path)

    
    return model

Qwen2_5OmniForConditionalGeneration.from_config = patched_from_config

device_map = {
    "thinker.model": "cuda", 
    "thinker.lm_head": "cuda", 
    "thinker.visual": "cpu",  
    "thinker.audio_tower": "cpu",  
    "talker": "cuda",  
    "token2wav": "cuda",  
}


# GPTQ MODEL
model = GPTQModel.load(
    model_path, 
    device_map=device_map, 
    torch_dtype=torch.float16,   
    attn_implementation="flash_attention_2"
)
processor = Qwen2_5OmniProcessor.from_pretrained(model_path)

def video_inference(video_path, prompt, sys_prompt):
    messages = [
        {"role": "system", "content": [
                {"type": "text", "text": sys_prompt},
            ]},
        {"role": "user", "content": [
                {"type": "video", "video": video_path},
            ]
        },
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    audios, images, videos = process_mm_info(messages, use_audio_in_video=True)
    inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True)
    inputs = inputs.to('cuda').to(model.dtype)
    

    output = model.generate(**inputs, use_audio_in_video=True, return_audio=True)
    text = processor.batch_decode(output[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    audio = output[2]
    return text, audio


video_path = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-Omni/draw.mp4"
system_prompt = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."

torch.cuda.reset_peak_memory_stats()
start = time.time()
response, audio  = video_inference(video_path, prompt=None, sys_prompt=system_prompt)
end = time.time()
peak_memory = torch.cuda.max_memory_allocated()

audio_file_path = "./output_audio_gptq.wav"
sf.write(
    audio_file_path,
    audio.reshape(-1).detach().cpu().numpy(),
    samplerate=24000,
)

print(response[0])
print(f"Total Inference Time: {end-start:.2f} s.")
print(f"Peak GPU Memory Used: {peak_memory / 1024 / 1024:.2f} MB")
