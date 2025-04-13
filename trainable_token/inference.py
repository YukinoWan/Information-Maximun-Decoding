import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, WhisperProcessor, WhisperModel
from model_components import FrameTokenizer, build_inputs_embeds  # 假设你把之前的组件放到 model_components.py 中

def load_components(device="cuda"):
    # 加载 Whisper 和 Qwen 模型（已冻结）
    whisper_model = WhisperModel.from_pretrained("openai/whisper-large-v3").to(device)
    whisper_model.eval()
    for p in whisper_model.parameters():
        p.requires_grad = False

    qwen_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-7B-Instruct").to(device)
    qwen_model.eval()
    for p in qwen_model.parameters():
        p.requires_grad = False

    # 加载 tokenizer 和 processor
    whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
    qwen_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")

    # 加载训练好的 FrameTokenizer（结构已知）
    frame_tokenizer = FrameTokenizer(encoder_dim=1280, token_dim=qwen_model.config.hidden_size)
    frame_tokenizer.load_state_dict(torch.load("frame_tokenizer.pt"))
    frame_tokenizer = frame_tokenizer.to(device)
    frame_tokenizer.eval()

    return frame_tokenizer, whisper_model, qwen_model, qwen_tokenizer, whisper_processor


def inference(audio_tensor, question_text, frame_tokenizer, whisper_model, qwen_model, tokenizer, processor, device="cuda"):
    frame_tokenizer.eval()
    whisper_model.eval()
    qwen_model.eval()

    with torch.no_grad():
        inputs = processor(audio_tensor, return_tensors="pt").to(device)
        encoder_outputs = whisper_model.encoder(**inputs).last_hidden_state
        decode_outputs = whisper_model.generate(inputs["input_features"], return_timestamps=True, output_hidden_states=False)

    whisper_texts = [x["text"] for x in decode_outputs]
    token_timestamps = [[(t["start"], t["end"]) for t in x["tokens"]] for x in decode_outputs]
    whisper_input_ids = [tokenizer(text, return_tensors='pt').input_ids[0].to(device) for text in whisper_texts]

    special_tokens = frame_tokenizer(encoder_outputs, token_timestamps)
    question_input_ids = tokenizer([question_text], return_tensors="pt", padding=True).input_ids.to(device)
    inputs_embeds = build_inputs_embeds(torch.stack(whisper_input_ids), special_tokens, question_input_ids, qwen_model)

    generated_ids = qwen_model.generate(inputs_embeds=inputs_embeds, max_new_tokens=50)
    output_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return output_text[0]