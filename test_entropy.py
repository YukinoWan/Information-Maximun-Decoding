from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import whisper

# 1. 加载模型和processor
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
processor = WhisperProcessor.from_pretrained("openai/whisper-small")

# 2. 准备音频输入（示例中是路径，你也可以用数组）
from datasets import load_dataset
import torchaudio

# ds = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
# sample = ds[0]["audio"]

sample = "data/MMAU/test-mini-audios/3fe64f3d-282c-4bc8-a753-68f8f6c35652.wav"
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

# 4. 编码器 forward（只跑一次）
with torch.no_grad():
    encoder_outputs = model.get_encoder()(input_features)

# 5. 初始化 decoder 输入
generated_ids = torch.tensor([[model.config.decoder_start_token_id]])  # e.g., [[50257]]

# 6. 开始一个 token 一个 token 地生成
max_new_tokens = 100
model.eval()
for _ in range(max_new_tokens):
    with torch.no_grad():
        outputs = model(
            input_features=input_features,
            decoder_input_ids=generated_ids,
            encoder_outputs=encoder_outputs,
            return_dict=True,
        )
        next_token_logits = outputs.logits[:, -1, :]  # 取出最后一个位置的 logits
        next_token_id = torch.argmax(next_token_logits, dim=-1)  # 贪婪解码

        # 停止条件
        if next_token_id.item() == processor.tokenizer.eos_token_id:
            break

        # 拼接到已生成序列中
        generated_ids = torch.cat([generated_ids, next_token_id.unsqueeze(-1)], dim=-1)

# 7. 解码成文本
decoded_text = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
print(decoded_text)
