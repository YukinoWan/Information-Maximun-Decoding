import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from train_decoder import compute_mc_loss, extract_answer, QAExampleDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, WhisperProcessor, WhisperForConditionalGeneration
from tqdm import tqdm


import re

import torchaudio
import json
import os




class FrameTokenizer(nn.Module):
    def __init__(self, encoder_dim=1280, embedding_dim=1024):
        super().__init__()
        # 先将音频特征映射到中间空间
        self.audio_encoder = nn.Sequential(
            nn.Linear(encoder_dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),
            nn.GELU()
        )
        
        # 从中间空间映射到LLM embedding space
        self.embedding_mapper = nn.Sequential(
            nn.Linear(1024, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.Tanh()  # 使用Tanh限制输出范围
        )
        
        # 确保所有参数可训练
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, encoder_hidden_states, token_timestamps):
        """
        Args:
            encoder_hidden_states: [batch_size, seq_len, encoder_dim]
            token_timestamps: List of token timestamps for each sample in batch
        Returns:
            token_embeddings: [batch_size, num_tokens, embedding_dim]
        """

        # 检查输入
        if torch.isnan(encoder_hidden_states).any() or torch.isinf(encoder_hidden_states).any():
            print("Input contains NaN or Inf values")
            return None
            
        batch_size = encoder_hidden_states.size(0)
        all_embeddings = []
        
        for b in range(batch_size):
            hidden = encoder_hidden_states[b]  # (seq_len, encoder_dim)
            batch_embeddings = []
            
            for start, end in token_timestamps[b]:
                # 获取对应时间段的hidden states
                if start >= hidden.size(0) or end > hidden.size(0):
                    print(f"Invalid timestamp: start={start}, end={end}, seq_len={hidden.size(0)}")
                    continue
                    
                time_slice = hidden[start:end]
                
                # 对时间段内的hidden states取平均
                if time_slice.size(0) > 0:
                    pooled = time_slice.mean(dim=0)
                else:
                    pooled = hidden[0]  # 使用第一个token作为fallback
                
                # 先将音频特征映射到中间空间
                intermediate = self.audio_encoder(pooled)
                
                # 从中间空间映射到LLM embedding space
                token_embedding = self.embedding_mapper(intermediate)
                
                batch_embeddings.append(token_embedding)
            
            if batch_embeddings:
                all_embeddings.append(torch.stack(batch_embeddings))
            else:
                # 如果没有有效的embeddings，使用第一个token
                first_token = self.embedding_mapper(self.audio_encoder(hidden[0]))
                all_embeddings.append(first_token.unsqueeze(0))
        
        # 将embeddings列表转换为tensor
        max_tokens = max(emb.size(0) for emb in all_embeddings)
        padded_embeddings = []
        
        for emb in all_embeddings:
            if emb.size(0) < max_tokens:
                padding = emb[-1].unsqueeze(0).repeat(max_tokens - emb.size(0), 1)
                padded_emb = torch.cat([emb, padding], dim=0)
            else:
                padded_emb = emb
            padded_embeddings.append(padded_emb)
        
        token_embeddings = torch.stack(padded_embeddings)
        
        # 检查输出
        if torch.isnan(token_embeddings).any() or torch.isinf(token_embeddings).any():
            print("Output contains NaN or Inf values")
            return None
            
        return token_embeddings





# def extract_answer_span(label_ids, tokenizer):
#     B, L = label_ids.shape
#     answer_positions = []
    
#     # Print the first few tokens in the vocabulary
#     # print("First 100 tokens in vocabulary:")
#     for i, (token, id) in enumerate(tokenizer.get_vocab().items()):
#         if i < 100:
#             print(f"{token}: {id}")
    
#     # Print the tokens in the label
#     # print("\nTokens in label:")
#     for i in range(B):
#         ids = label_ids[i].tolist()
#         tokens = tokenizer.convert_ids_to_tokens(ids)
#         # print(f"Label tokens: {tokens}")
        
#         # Try to find the answer markers
#         start_idx = None
#         end_idx = None
#         for j, token in enumerate(tokens):
#             if token == "<answer>":
#                 start_idx = j
#             elif token == "</answer>":
#                 end_idx = j
        
#         if start_idx is not None and end_idx is not None:
#             # print(f"Found answer span: start={start_idx}, end={end_idx}")
#             answer_positions.append((start_idx, end_idx))
#         else:
#             print("Warning: Could not find answer span markers")
#             answer_positions.append((0, 0))
    
#     return answer_positions



class Trainer:
    def __init__(self, frame_tokenizer, whisper_model, qwen_model, qwen_tokenizer, whisper_processor, optimizer, device="cuda"):
        self.frame_tokenizer = frame_tokenizer
        self.whisper_model = whisper_model
        self.qwen_model = qwen_model
        self.tokenizer = qwen_tokenizer
        self.whisper_processor = whisper_processor
        self.optimizer = optimizer
        self.device = device


    def training_step(self, batch):

        # 获取音频路径、问题和答案
        audio_tensor = batch['audio'].to(self.device)
        question_text = batch['question']
        correct_answer_text = batch['answer']
        
        # 获取encoder outputs和transcription
        with torch.no_grad():
            inputs = self.whisper_processor(audio_tensor.cpu().numpy(), sampling_rate=16000, return_tensors="pt").to(self.device)
            encoder_outputs = self.whisper_model.model.encoder(**inputs).last_hidden_state
            
            # 获取transcription
            decode_outputs = self.whisper_model.generate(
                inputs["input_features"],
                return_timestamps=True,
                output_hidden_states=False,
                language="en"
            )
            whisper_texts = self.whisper_processor.batch_decode(decode_outputs, skip_special_tokens=True)
            
            # 创建timestamps
            num_words = len(whisper_texts[0].split())
            if num_words == 0:
                return None, 0
            token_timestamps = [[(i, i+1) for i in range(num_words)]]

        # 获取token embeddings
        token_embeddings = self.frame_tokenizer(encoder_outputs, token_timestamps)

        # 第二步：使用FrameTokenizer生成token embeddings


        # 第三步：构造输入prompt和目标输出
        # 输入prompt
        prompt = (
            f"Here is an audio transcription with hints (e.g.,sound, music, etc.) at each time stamp: {whisper_texts[0]} \nHints: \n"
            f"{question_text}\n"
            f"Please provide your answer in the format: The answer is <answer>your_answer</answer>"
        )
        
        # 目标输出
        target = f"The answer is <answer>{correct_answer_text}</answer>"

        messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},    # your input
        {"role": "assistant", "content": target}     # your expected output
    ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_tensors="pt"
        ).to(self.device)
        # print(f"Messages: {messages}")
        # # 第四步：获取prompt的embeddings
        # prompt_inputs = self.tokenizer(
        #     [prompt],
        #     return_tensors="pt",
        #     padding=True,
        #     add_special_tokens=True,
        #     truncation=True,
        #     max_length=512
        # ).to(self.device)
        # print(f"Token embeddings: {token_embeddings.shape}")

        
        # 获取目标输出的token ids
        # target_inputs = self.tokenizer(
        #     [target],
        #     add_special_tokens=False
        # ).input_ids
        whisper_ids = self.tokenizer(
            "Hints:",
            add_special_tokens=False
        ).input_ids

        
        instr_ids = self.tokenizer.apply_chat_template(messages[:-1], tokenize=True, return_tensors="pt").to(self.device)


        def find_subsequence(source, target):
            for i in range(len(source) - len(target) + 1):
                if source[i:i+len(target)] == target:
                    return i, i+len(target)
            return -1, -1

        # print(f"Message: {messages}")
        # print(f"Whisper texts: {whisper_texts}")
        # print(f"Instr ids: {instr_ids[0].tolist()}")
        # print(f"Whisper ids: {whisper_ids}")
        start, end = find_subsequence(instr_ids[0].tolist(), whisper_ids)
        # print(f"Start: {start}, End: {end}")
        # assert False

        
        n_instr = instr_ids.shape[1] + token_embeddings.shape[1]
        labels = prompt.clone()
        left = labels[:, :end]                           # tensor (1, end)
        middle = torch.full((1, token_embeddings.shape[1]), -100, dtype=labels.dtype, device=labels.device)  # tensor (1, N)
        right = labels[:, end:] 
        labels = torch.cat([left, middle, right], dim=1)
        labels[:, :n_instr] = -100

        # 获取prompt embeddings
        prompt_embeddings = self.qwen_model.get_input_embeddings()(prompt)

        # instr_embeddings = self.qwen_model.get_input_embeddings()(instr_ids)

        # 转换为tensor
        # input_embeddings = torch.stack(new_embeddings).unsqueeze(0)
        # print(f"Prompt embeddings: {prompt_embeddings.shape}")
        # print(f"Token embeddings: {token_embeddings.shape}")
        before_token = prompt_embeddings[:, :end,:]
        after_token = prompt_embeddings[:, end:,:]
        input_embeddings = torch.cat([before_token, token_embeddings, after_token], dim=1)
        # 清理不需要的tensors
        # del prompt_embeddings, new_embeddings
        torch.cuda.empty_cache()

        # 计算loss
        with torch.cuda.amp.autocast():
            loss, tmp_acc = compute_mc_loss(
                self.qwen_model,
                input_embeddings,
                labels,
                self.tokenizer
            )


        
        return loss, tmp_acc




def train_loop(dataloader, trainer, epochs=10):
    for epoch in range(epochs):
        total_loss = 0
        processed_samples = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

        total_acc = 0
        
        for batch in pbar:
            # 训练步骤
            trainer.optimizer.zero_grad()
            loss, acc = trainer.training_step(batch)
            
            if loss is not None:
                loss.backward()
                trainer.optimizer.step()
                total_loss += loss.item()
                processed_samples += 1
                total_acc += acc
            # 更新进度条
            if processed_samples > 0:
                avg_loss = total_loss / processed_samples
                pbar.set_postfix({"loss": f"{avg_loss:.4f}", "acc": f"{total_acc/processed_samples:.4f}"})
        
        # 保存模型
        if processed_samples > 0:
            avg_loss = total_loss / processed_samples
            print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}, Average Acc: {total_acc/processed_samples:.4f}")
            save_tokenizer(trainer.frame_tokenizer, f"output/mmau_frame_tokenizer_encoder_epoch_{epoch+1}.pt")

def save_tokenizer(frame_tokenizer, path="frame_tokenizer.pt"):
    torch.save(frame_tokenizer.state_dict(), path)

# === Example usage ===
if __name__ == "__main__":
    # Set environment variable to avoid memory fragmentation
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Loading models...")
    # Load models
    whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
    whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3")
    whisper_model = whisper_model.to(device)

    qwen_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    qwen_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-7B-Instruct")
    qwen_model = qwen_model.to(device)
    qwen_model.eval()
    
    # Keep parameters frozen but allow gradient computation
    for param in qwen_model.parameters():
        param.requires_grad = False

    whisper_model.eval()
    for param in whisper_model.parameters():
        param.requires_grad = False
    
    print("Creating dataset and dataloader...")
    # Create dataset and dataloader
    dataset = QAExampleDataset("/home/zhen/Information-Maximun-Decoding/data/MMAU/train_final_sub.json")
    dataloader = DataLoader(
        dataset, 
        batch_size=1, 
        shuffle=True,
        num_workers=0
    )
    
    print("Creating model and optimizer...")
    # Create model and optimizer
    # 获取Qwen2的embedding维度
    embedding_dim = qwen_model.get_input_embeddings().weight.size(1)
    frame_tokenizer = FrameTokenizer(encoder_dim=1280, embedding_dim=embedding_dim)
    frame_tokenizer = frame_tokenizer.to(device)
    
    # 创建优化器，只优化frame tokenizer
    optimizer = torch.optim.AdamW([
        {"params": frame_tokenizer.parameters(), "lr": 1e-4}
    ])
    
    # Create trainer
    trainer = Trainer(
        frame_tokenizer, 
        whisper_model, 
        qwen_model, 
        qwen_tokenizer, 
        whisper_processor, 
        optimizer,
        device=device
    )
    
    # Clear CUDA cache before training
    torch.cuda.empty_cache()
    
    print("Starting training...")
    # Train
    train_loop(dataloader, trainer, epochs=10)
    
    print("Saving model...")
    # Save model
    
    print("Training completed!")

