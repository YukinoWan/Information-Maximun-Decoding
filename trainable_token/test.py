import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, WhisperProcessor, WhisperModel
import torchaudio
import json

class FrameTokenizer(nn.Module):
    def __init__(self, encoder_dim: int, token_dim: int):
        super().__init__()
        self.mapper = nn.Sequential(
            nn.Linear(encoder_dim, token_dim),
            nn.ReLU(),
            nn.LayerNorm(token_dim)
        )

    def forward(self, encoder_hidden_states, token_timestamps, frame_rate=50.0):
        special_tokens = []
        for b in range(encoder_hidden_states.size(0)):
            hidden = encoder_hidden_states[b]  # (T_enc, D_enc)
            tokens = []
            for (start, end) in token_timestamps[b]:
                s_idx = int(start * frame_rate)
                e_idx = max(int(end * frame_rate), s_idx + 1)
                pooled = hidden[s_idx:e_idx].mean(dim=0)
                token = self.mapper(pooled)
                tokens.append(token)
            special_tokens.append(torch.stack(tokens))  # (T_token, D)
        return torch.stack(special_tokens)  # (B, T_token, D)


def build_inputs_embeds(whisper_tokens, special_tokens, question_input_ids, llm_model):
    input_embeds = []
    for i in range(whisper_tokens.size(0)):
        text_emb = llm_model.get_input_embeddings()(whisper_tokens[i])  # (T, D)
        special = special_tokens[i]  # (T, D)
        interleaved = torch.stack([text_emb, special], dim=1).reshape(-1, text_emb.size(-1))
        q_embed = llm_model.get_input_embeddings()(question_input_ids[i])  # (Q, D)
        full_embed = torch.cat([interleaved, q_embed], dim=0)
        input_embeds.append(full_embed)
    return torch.stack(input_embeds)  # (B, T+Q, D)


def extract_answer_span(label_ids, tokenizer):
    B, L = label_ids.shape
    answer_positions = []
    for i in range(B):
        ids = label_ids[i].tolist()
        start_tok = tokenizer.convert_tokens_to_ids("<answer>")
        end_tok = tokenizer.convert_tokens_to_ids("</answer>")
        try:
            start = ids.index(start_tok) + 1
            end = ids.index(end_tok)
        except ValueError:
            start, end = 0, 0
        answer_positions.append((start, end))
    return answer_positions


def compute_mc_loss(llm_model, inputs_embeds, labels, answer_spans):
    outputs = llm_model(inputs_embeds=inputs_embeds, labels=labels)
    logits = outputs.logits
    losses = []
    for i, (start, end) in enumerate(answer_spans):
        for pos in range(start, end):
            losses.append(F.cross_entropy(logits[i, pos], labels[i, pos]))
    return sum(losses) / len(losses)


class Trainer:
    def __init__(self, model, whisper, qwen, tokenizer, whisper_proc, optimizer, device="cuda"):
        self.frame_tokenizer = model.to(device)
        self.whisper_model = whisper.to(device)
        self.qwen_model = qwen.to(device)
        self.tokenizer = tokenizer
        self.processor = whisper_proc
        self.optimizer = optimizer
        self.device = device

    def training_step(self, audio_tensor, question_text, correct_answer_text):
        with torch.no_grad():
            inputs = self.processor(audio_tensor, return_tensors="pt").to(self.device)
            encoder_outputs = self.whisper_model.encoder(**inputs).last_hidden_state
            decode_outputs = self.whisper_model.generate(inputs["input_features"], return_timestamps=True, output_hidden_states=False)

        whisper_texts = [x["text"] for x in decode_outputs]
        token_timestamps = [[(t["start"], t["end"]) for t in x["tokens"]] for x in decode_outputs]
        whisper_input_ids = [self.tokenizer(text, return_tensors='pt').input_ids[0].to(self.device) for text in whisper_texts]

        special_tokens = self.frame_tokenizer(encoder_outputs, token_timestamps)
        question_input_ids = self.tokenizer([question_text], return_tensors="pt", padding=True).input_ids.to(self.device)
        inputs_embeds = build_inputs_embeds(torch.stack(whisper_input_ids), special_tokens, question_input_ids, self.qwen_model)

        answer_prompt = f"<answer>{correct_answer_text}</answer>"
        label_ids = self.tokenizer([answer_prompt], return_tensors="pt", padding=True).input_ids.to(self.device)
        answer_spans = extract_answer_span(label_ids, self.tokenizer)

        loss = compute_mc_loss(self.qwen_model, inputs_embeds, label_ids, answer_spans)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item()


class QAExampleDataset(Dataset):
    def __init__(self, jsonl_path):
        self.data = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        audio_tensor, _ = torchaudio.load(item['audio_path'])
        return {
            'audio': audio_tensor,
            'question': item['question'],
            'answer': item['answer']
        }


def train_loop(dataloader, trainer, epochs=1):
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            audio = batch['audio'].to(trainer.device)
            question = batch['question']
            answer = batch['answer']
            loss = trainer.training_step(audio, question, answer)
            total_loss += loss
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}, Avg Loss: {avg_loss:.4f}")

def save_tokenizer(frame_tokenizer, path="frame_tokenizer.pt"):
    torch.save(frame_tokenizer.state_dict(), path)

# === Example usage ===
if __name__ == "__main__":
    whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
    whisper_model = WhisperModel.from_pretrained("openai/whisper-large-v3")
    qwen_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    qwen_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-7B-Instruct")
    qwen_model.eval()
    for param in qwen_model.parameters():
        param.requires_grad = False

    whisper_model.eval()
    for param in whisper_model.parameters():
        param.requires_grad = False
    dataset = QAExampleDataset("data/train.jsonl")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    tokenizer_dim = qwen_model.config.hidden_size
    frame_tokenizer = FrameTokenizer(encoder_dim=1280, token_dim=tokenizer_dim)  # whisper-large-v3 hidden dim = 1280
    optimizer = torch.optim.AdamW(frame_tokenizer.parameters(), lr=1e-4)
    trainer = Trainer(frame_tokenizer, whisper_model, qwen_model, qwen_tokenizer, whisper_processor, optimizer)
    train_loop(dataloader, trainer, epochs=3)
    save_tokenizer(frame_tokenizer, "frame_tokenizer.pt")

