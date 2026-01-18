# 实时数字形象生成

## Thinker + FlashTalk 联动（多进程 IPC）

该模式在同一入口脚本中启动 Thinker（GPU0）与 FlashTalk 多卡推理（GPU1-7），通过 `multiprocessing` 队列进行进程间通信，不依赖文件队列或 HTTP 服务。

### 模型准备（示例路径）
- Thinker: `models/Qwen2.5-Omni-7B`
- FlashTalk: `models/SoulX-FlashTalk-14B`
- Wav2Vec: `models/chinese-wav2vec2-base`

### 启动命令
```bash
python src/thinker/thinker_flash_talk_demo.py \
  --thinker_ckpt models/Qwen2.5-Omni-7B \
  --thinker_device cuda:0 \
  --start_flashtalk_worker \
  --flashtalk_ckpt models/SoulX-FlashTalk-14B \
  --flashtalk_wav2vec models/chinese-wav2vec2-base \
  --flashtalk_gpus 1,2,3,4,5,6,7 \
  --flashtalk_dist_port 29501
```

### 说明
- Thinker 固定使用 `cuda:0`，FlashTalk 使用 `1-7`；两者不可重叠。
- FlashTalk 条件图可在 Gradio 界面选择（默认使用 `--flashtalk_cond_image`）。
