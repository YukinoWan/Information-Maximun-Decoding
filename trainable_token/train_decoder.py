import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from peft import get_peft_config, get_peft_model, IA3Config, TaskType

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


def extract_answer(output_str):
    answer_pattern = r"<answer>(.*?)</answer>"
    match = re.search(answer_pattern, output_str)

    if match:
        return match.group(1)
    return output_str


class FrameTokenizer(nn.Module):
    def __init__(self, hidden_size=1024, embedding_dim=1024):
        super().__init__()
        self.audio_encoder = nn.Sequential(
            nn.Linear(hidden_size, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),
            nn.GELU()
        )
        
        self.embedding_mapper = nn.Sequential(
            nn.Linear(1024, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.Tanh()
        )
        
        # 确保所有参数可训练
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, decoder_hidden_states):
        """
        Args:
            decoder_hidden_states: [batch_size, seq_len, hidden_size]
        Returns:
            token_embeddings: [batch_size, seq_len, embedding_dim]
        """
        try:
            # 检查输入
            if torch.isnan(decoder_hidden_states).any() or torch.isinf(decoder_hidden_states).any():
                print("Input contains NaN or Inf values")
                return None
                
            batch_size, seq_len, hidden_size = decoder_hidden_states.shape
            
            # 1. 将每个token的hidden state映射到中间空间
            projected_states = self.audio_encoder(decoder_hidden_states)
            
            # 2. 将中间表示映射到LLM的embedding空间
            token_embeddings = self.embedding_mapper(projected_states)
            
            # 检查输出
            if torch.isnan(token_embeddings).any() or torch.isinf(token_embeddings).any():
                print("Output contains NaN or Inf values")
                return None
                
            return token_embeddings
            
        except Exception as e:
            print(f"Error in FrameTokenizer forward: {str(e)}")
            return None




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


def compute_mc_loss(llm_model, input_embeds, labels, tokenizer):
    """
    计算SFT loss

    """
    # 使用input embeddings进行生成
    outputs = llm_model(inputs_embeds=input_embeds, labels=labels, return_dict=True)
    # logits = outputs.logits
    loss = outputs.loss
    logits = outputs.logits
    
    # 获取生成的文本（使用argmax）
    pred_ids = torch.argmax(logits, dim=-1)[0]
    labels = labels[0]
    pred_text = tokenizer.decode(pred_ids, skip_special_tokens=False)
    label_text = tokenizer.decode(labels[labels != -100], skip_special_tokens=False)
    answer_span = extract_answer(pred_text)
    label_span = extract_answer(label_text)

    print(f"pred_text: {pred_text}")
    print(f"label_text: {label_text}")

    if answer_span == label_span:
        acc = 1
    else:
        acc = 0
    print(f"Generated text: {answer_span}")
    print(f"Label text: {label_span}")
    
    # 计算生成的loss
    # shift_logits = logits[..., :-1, :].contiguous()
    # shift_labels = target_ids[..., 1:].contiguous()
    
    # 计算交叉熵loss
    # loss = F.cross_entropy(
    #     shift_logits.reshape(-1, shift_logits.size(-1)),
    #     shift_labels.reshape(-1),
    #     ignore_index=tokenizer.pad_token_id
    # )
    
    print(f"SFT loss: {loss.item():.4f}")
    
    # Clear unnecessary tensors
    del outputs, logits, pred_ids, labels
    torch.cuda.empty_cache()
    
    return loss, acc


class Trainer:
    def __init__(self, model, whisper, qwen, tokenizer, whisper_proc, optimizer, device="cuda"):
        self.frame_tokenizer = model.to(device)
        self.whisper_model = whisper.to(device)
        self.qwen_model = qwen.to(device)
        self.tokenizer = tokenizer
        self.processor = whisper_proc
        self.optimizer = optimizer
        self.device = device
        
        # 启用梯度检查点
        self.qwen_model.gradient_checkpointing_enable()
        
        # 保存初始参数用于比较
        self.initial_params = {name: param.clone().detach() for name, param in self.frame_tokenizer.named_parameters()}

    def training_step(self, audio_tensor, question_text, correct_answer_text):
        self.qwen_model.eval()
        self.whisper_model.eval()
        self.frame_tokenizer.train()

        # Clear CUDA cache
        torch.cuda.empty_cache()

        # 第一步：获取音频转录和encoder输出
        with torch.no_grad():
            # 处理音频输入
            inputs = self.processor(audio_tensor.cpu().numpy(), sampling_rate=16000, return_tensors="pt").to(self.device)
            # encoder_outputs = self.whisper_model.model.encoder(**inputs).last_hidden_state
            decoder_input_ids = torch.tensor([[self.whisper_model.config.decoder_start_token_id]]).to(self.device)
            outputs = self.whisper_model(
                input_features=inputs.input_features,
                decoder_input_ids=decoder_input_ids,
                output_hidden_states=True,
                return_dict=True
            )
            # 获取音频转录
            decode_outputs = self.whisper_model.generate(
                inputs["input_features"],
                return_timestamps=True,
                output_hidden_states=False,
                language="en"
            )
            whisper_texts = self.processor.batch_decode(decode_outputs, skip_special_tokens=True)
            decoder_hidden_states = outputs.decoder_hidden_states[-1]
            # 创建时间戳
            num_words = len(whisper_texts[0].split())
            token_timestamps = [[(i, i+1) for i in range(num_words)]]
            
            # 清理不需要的tensors
            del inputs, decode_outputs
            torch.cuda.empty_cache()

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

        token_embeddings = self.frame_tokenizer(decoder_hidden_states)
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

        # 清理encoder输出
        del decoder_hidden_states
        torch.cuda.empty_cache()

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

        # 第七步：计算SFT loss
        with torch.cuda.amp.autocast():
            loss, tmp_acc = compute_mc_loss(
                self.qwen_model,
                input_embeddings,
                labels,
                self.tokenizer
            )

        # 反向传播和优化
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        # 清理内存
        del input_embeddings, token_embeddings
        torch.cuda.empty_cache()

        return loss.item(), tmp_acc


class QAExampleDataset(Dataset):
    def __init__(self, jsonl_path):
        self.data = []
        with open(jsonl_path, 'r') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # print("Trying to load:", item["audio"])
        try:
            audio_tensor, _ = torchaudio.load(item["audio"])  # (channels, length)
        except Exception as e:
            print(f"Error in loading audio: {e}")
            return self.__getitem__((idx + 1) % len(self))
        # 👉 转成 float32 + 单通道
        if audio_tensor.dim() == 2:  # stereo
            audio_tensor = audio_tensor.mean(dim=0)  # -> mono
        audio_tensor = audio_tensor.float()  # make sure it's float32

        return {
            "audio": audio_tensor.squeeze(0),  # shape (samples,)
            "question": item["question"],
            "answer": item["answer"]
        }



def train_loop(dataloader, trainer, epochs=1):
    for epoch in range(epochs):
        total_loss = 0
        # Create progress bar for batches
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        total_acc = 0
        
        for i, batch in enumerate(pbar):
            if batch is None:
                continue
            # Clear CUDA cache at the start of each batch
            torch.cuda.empty_cache()
            
            audio = batch['audio'].to(trainer.device)
            question = batch['question']
            answer = batch['answer']
            try:
                loss, acc = trainer.training_step(audio, question, answer)
            except Exception as e:
                print(f"Error in training_step: {e}")
                continue
            total_loss += loss
            total_acc += acc
            
            # Update progress bar description
            pbar.set_postfix({
                'loss': f'{loss:.4f}',
                'avg_loss': f'{total_loss/(i+1):.4f}',
                'acc': f'{total_acc/(i+1):.4f}'
            })
            
            # Clear CUDA cache after each batch
            torch.cuda.empty_cache()
            
            # Force garbage collection
            import gc
            gc.collect()
            
        avg_loss = total_loss / len(dataloader)
        print(f"\nEpoch {epoch+1}, Avg Loss: {avg_loss:.4f}")
        if True:
            trainer.whisper_model.save_pretrained(f"output/decoder_only/MMAU_encoder_adapter_{epoch}.pt")

        save_tokenizer(trainer.frame_tokenizer, f"output/decoder_only/MMAU_encoder_adapter_frame_decoder_tokenizer_{epoch}.pt")

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
    if True:
        target_modules = []
        for name, module in whisper_model.model.encoder.named_modules():
            print(name)
            if "k_proj" in name or "v_proj" in name:
                target_modules.append(name)
        peft_config = IA3Config(
            target_modules=target_modules, feedforward_modules=[]
        )
        whisper_model = get_peft_model(whisper_model, peft_config)
        whisper_model.print_trainable_parameters()
        whisper_model.train()
    else:
        whisper_model.eval()
        for param in whisper_model.parameters():
            param.requires_grad = False
    whisper_model = whisper_model.to(device)

    qwen_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    qwen_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-7B-Instruct", attn_implementation="flash_attention_2", device_map="auto")
    # qwen_model = qwen_model.to(device)
    qwen_model.eval()
    
    # Keep parameters frozen but allow gradient computation
    for param in qwen_model.parameters():
        param.requires_grad = False
    # qwen_model.requires_grad_(True)  # Enable gradient computation for the model

  
    
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
    # 获取Whisper decoder的hidden size
    decoder_hidden_size = whisper_model.config.d_model
    frame_tokenizer = FrameTokenizer(hidden_size=decoder_hidden_size, embedding_dim=embedding_dim)

    frame_tokenizer = frame_tokenizer.to(device)
    total_params = sum(p.numel() for p in frame_tokenizer.parameters() if p.requires_grad)

    print(f"Trainable parameters: {total_params}")
    # assert False
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

