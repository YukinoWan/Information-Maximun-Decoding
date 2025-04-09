from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import whisper

# 1. 加载模型和processor
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3")
processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")

# 2. 准备音频输入（示例中是路径，你也可以用数组）
from datasets import load_dataset
import torchaudio

# ds = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
# sample = ds[0]["audio"]

sample = "data/MMAU/test-mini-audios/6aee68bf-6629-442b-981d-ae8195597c8e.wav"
model2 = whisper.load_model("large-v3")
text = model2.transcribe(sample, word_timestamps=True, temperature=0.8)["text"]
print(text)

waveform, sr = torchaudio.load(sample)
if sr != 16000:
    waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
    sr = 16000
if waveform.shape[0] > 1:
    waveform = waveform.mean(dim=0)  # 将立体声变为单声道
else:
    waveform = waveform.squeeze(0)
# 3. 获取模型输入
inputs = processor(waveform, sampling_rate=sr, return_tensors="pt")
input_features = inputs["input_features"]

# # 4. 编码器 forward（只跑一次）
# with torch.no_grad():
#     encoder_outputs = model.get_encoder()(input_features)

# # 5. 初始化 decoder 输入
# generated_ids = torch.tensor([[model.config.decoder_start_token_id]])  # e.g., [[50257]]

# # 6. 开始一个 token 一个 token 地生成
# max_new_tokens = 100
# model.eval()
# for _ in range(max_new_tokens):
#     with torch.no_grad():
#         outputs = model(
#             input_features=input_features,
#             decoder_input_ids=generated_ids,
#             encoder_outputs=encoder_outputs,
#             return_dict=True,
#         )
#         next_token_logits = outputs.logits[:, -1, :]  # 取出最后一个位置的 logits
#         next_token_id = torch.argmax(next_token_logits, dim=-1)  # 贪婪解码

#         # 停止条件
#         if next_token_id.item() == processor.tokenizer.eos_token_id:
#             break

#         # 拼接到已生成序列中
#         generated_ids = torch.cat([generated_ids, next_token_id.unsqueeze(-1)], dim=-1)

# # 7. 解码成文本
# decoded_text = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
# print(decoded_text)

import torch
import torch.nn.functional as F
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset
import torchaudio


# 3. 提取特征
inputs = processor(waveform.squeeze(0), sampling_rate=16000, return_tensors="pt")
input_features = inputs["input_features"]

# 4. 编码器输出
with torch.no_grad():
    encoder_outputs = model.get_encoder()(input_features)

# 5. 初始化
generated_ids = torch.tensor([[model.config.decoder_start_token_id]])  # decoder 开始标志
eos_token_id = processor.tokenizer.eos_token_id
max_new_tokens = 50
entropy_threshold = 1.2  # 你可以调这个阈值

# 6. token-by-token 解码
model.eval()
all_output_tokens = []  # 每一步的候选 token

for _ in range(max_new_tokens):
    with torch.no_grad():
        outputs = model(
            input_features=input_features,
            decoder_input_ids=generated_ids,
            encoder_outputs=encoder_outputs,
            return_dict=True,
        )
        logits = outputs.logits[:, -1, :]  # 只看最后一位
        probs = F.softmax(logits, dim=-1)

        # 计算熵
        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1).item()

        if entropy < entropy_threshold:
            # 置信度高 → greedy decode
            next_token = torch.argmax(probs, dim=-1)
            generated_ids = torch.cat([generated_ids, next_token.unsqueeze(-1)], dim=-1)

            all_output_tokens.append([next_token.item()])  # 用列表包装，便于统一结构

            if next_token.item() == eos_token_id:
                break
        else:
            # 置信度低 → 取 top-3
            topk_probs, topk_indices = torch.topk(probs, k=3, dim=-1)
            topk_token_ids = topk_indices[0].tolist()

            # 暂时先添加 top-1 到 decode（后续你也可以尝试beam search或并行路径）
            selected_token = topk_indices[0, 0].unsqueeze(0)
            generated_ids = torch.cat([generated_ids, selected_token.unsqueeze(0)], dim=-1)

            all_output_tokens.append(topk_token_ids)

# 7. 打印每一步的结果（支持多个候选 token）
for i, token_ids in enumerate(all_output_tokens):
    tokens = [processor.tokenizer.decode([tid]) for tid in token_ids]
    print(f"Step {i+1}: {' | '.join(tokens)}")

