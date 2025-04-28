import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, WhisperProcessor, WhisperForConditionalGeneration
from train_decoder import QAExampleDataset, extract_answer
from train_encoder import FrameTokenizer
import re
import argparse
import json
from tqdm import tqdm

def load_models(device="cuda"):
    # Load models
    whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
    whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3")
    whisper_model = whisper_model.to(device)
    whisper_model.eval()

    qwen_tokenizer = AutoTokenizer.from_pretrained("/home/zhen/internal_llm/internal_llm")
    qwen_model = AutoModelForCausalLM.from_pretrained("/home/zhen/internal_llm/internal_llm", device_map="auto")
    qwen_model = qwen_model.to(device)
    qwen_model.eval()

    # Load frame tokenizer
    embedding_dim = qwen_model.get_input_embeddings().weight.size(1)
    decoder_hidden_size = whisper_model.config.d_model
    frame_tokenizer = FrameTokenizer(encoder_dim=1280, embedding_dim=embedding_dim)
    frame_tokenizer.load_state_dict(torch.load("./output/Rllm_decoder_only/MMAU_frame_decoder_tokenizer_0.pt"))
    frame_tokenizer = frame_tokenizer.to(device)
    frame_tokenizer.eval()

    return whisper_processor, whisper_model, qwen_tokenizer, qwen_model, frame_tokenizer

def process_audio(audio_path, whisper_processor, whisper_model, frame_tokenizer, device="cuda"):
    try:
        # Load and process audio
        audio_tensor, _ = torchaudio.load(audio_path)
        if audio_tensor.dim() == 2:  # stereo
            audio_tensor = audio_tensor.mean(dim=0)  # -> mono
        audio_tensor = audio_tensor.float()

        # Get decoder outputs and transcription
        with torch.no_grad():
            inputs = whisper_processor(audio_tensor.cpu().numpy(), sampling_rate=16000, return_tensors="pt").to(device)
            encoder_outputs = whisper_model.model.encoder(**inputs).last_hidden_state
            
            # 获取transcription
            decode_outputs = whisper_model.generate(
                inputs["input_features"],
                return_timestamps=True,
                output_hidden_states=False,
                language="en"
            )
            
            whisper_texts = whisper_processor.batch_decode(decode_outputs, skip_special_tokens=True)
            
            # 创建timestamps
            num_words = len(whisper_texts[0].split())
            if num_words == 0:
                return None, 0
            token_timestamps = [[(i, i+1) for i in range(num_words)]]

        # 获取token embeddings
        token_embeddings = frame_tokenizer(encoder_outputs, token_timestamps)



        return whisper_texts[0], token_embeddings

    except Exception as e:
        print(f"Error in process_audio: {str(e)}")
        raise

def generate_answer(question, whisper_text, token_embeddings, qwen_tokenizer, qwen_model, device="cuda"):
    try:
        # Construct prompt
        prompt = (
            f"Here is an audio transcription with hints (e.g.,sound, music, etc.) at each time stamp: {whisper_text} \nHints: \n"
            f"{question}\n"
            f"Please provide your answer in the format: The answer is <answer>your_answer</answer>"
            # f"Please provide your answer in the format: The answer is <answer>your_answer</answer>"
        )

        # Prepare messages for chat template
        messages = [
            # {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "<think>\n"}
        ]

        # Get instruction embeddings
        instr_ids = qwen_tokenizer.apply_chat_template(messages, tokenize=True, return_tensors="pt").to(device)
        instr_embeddings = qwen_model.get_input_embeddings()(instr_ids)

        # Find position to insert token embeddings
        whisper_ids = qwen_tokenizer("Hints:", add_special_tokens=False).input_ids
        start, end = -1, -1
        for i in range(len(instr_ids[0]) - len(whisper_ids) + 1):
            if instr_ids[0][i:i+len(whisper_ids)].tolist() == whisper_ids:
                start, end = i, i + len(whisper_ids)
                break

        if start == -1 or end == -1:
            raise ValueError("Could not find position to insert token embeddings")

        # Insert token embeddings
        before_token = instr_embeddings[:, :end, :]
        after_token = instr_embeddings[:, end:, :]
        
        # Check dimensions
        if token_embeddings.size(2) != before_token.size(2):
            raise ValueError(f"Dimension mismatch: token_embeddings={token_embeddings.size(2)}, before_token={before_token.size(2)}")
            
        # input_embeddings = torch.cat([before_token, token_embeddings, after_token], dim=1)
        input_embeddings = instr_embeddings
        # Clear memory
        # del instr_ids, instr_embeddings, before_token, after_token
        torch.cuda.empty_cache()

        # Generate answer
        with torch.no_grad():
            outputs = qwen_model.generate(
                inputs_embeds=input_embeddings,
                max_new_tokens=2048,
                do_sample=False,
                temperature=0.0,
                top_p=0.9
            )
            generated_text = qwen_tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Clear memory
        del outputs, input_embeddings
        torch.cuda.empty_cache()

        return generated_text

    except Exception as e:
        print(f"Error in generate_answer: {str(e)}")
        raise

def extract_answer(output_str):
    answer_pattern = r"<answer>(.*?)</answer>"
    match = re.search(answer_pattern, output_str)
    if match:
        return match.group(1)
    return output_str

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="/home/zhen/Information-Maximun-Decoding/data/MMAU/test_final_sub.json")
    parser.add_argument("--output_path", type=str, default="evaluation_output/inference_reasoning_llm_results_0.json")
    args = parser.parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load models
    whisper_processor, whisper_model, qwen_tokenizer, qwen_model, frame_tokenizer = load_models(device)
    
    # Load data
    with open(args.data_path, 'r') as f:
        data = json.load(f)
    
    results = []
    # Process each example
    correct = 0
    total = 0

    output_data = []

    for example in tqdm(data, desc="Processing examples"):
        # 获取音频路径、问题和答案
        audio_path = example['audio']
        question = example['question']
        ground_truth = example['answer']
        
        try:
            # Clear CUDA cache before processing each example
            torch.cuda.empty_cache()
            
            # Process audio
            whisper_text, token_embeddings = process_audio(
                audio_path, 
                whisper_processor, 
                whisper_model, 
                frame_tokenizer, 
                device
            )
            
            # Generate answer
            generated_text = generate_answer(
                question,
                whisper_text,
                token_embeddings,
                qwen_tokenizer,
                qwen_model,
                device
            )
            
            # Extract answer and label
            answer = extract_answer(generated_text.split("</think>")[1])
            label = extract_answer(ground_truth)
            
            # Calculate accuracy
            if label.lower() in answer.lower():  # 忽略大小写比较
                acc = 1
            else:
                acc = 0
            
            # Save results
            results.append({
                'audio': audio_path,
                'question': question,
                'answer': ground_truth,
                'transcription': whisper_text,
                'generated_text': generated_text,
                'model_prediction': answer,
                'is_correct': acc
            })
            
            # Print progress
            total += 1
            correct += acc
            print(f"\nProcessing {audio_path}")
            print(f"Question: {question}")
            print(f"Ground truth: {ground_truth}")
            print(f"Generated answer: {answer}")
            print(f"Current accuracy: {correct/total:.2%}")
            print("--------------------------------")

            # Clear memory after each example
            del whisper_text, token_embeddings, generated_text, answer, label
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error processing {audio_path}: {str(e)}")
            print(f"Generated text: {generated_text}")
            continue

    # Save all results
    with open(args.output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nFinal accuracy: {correct/total:.2%}")
    print(f"Results saved to {args.output_path}")

if __name__ == "__main__":
    # Set CUDA_LAUNCH_BLOCKING for better error messages
    
    main()