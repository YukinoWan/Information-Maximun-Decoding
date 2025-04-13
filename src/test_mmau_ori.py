import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Optional
import sys
# import whisper

from datetime import timedelta
from difflib import SequenceMatcher
from collections import defaultdict
from transformers import WhisperProcessor, WhisperForConditionalGeneration


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn.functional as F
import torchaudio
import transformers
from tqdm import tqdm
from transformers import AutoProcessor, HfArgumentParser, Qwen2AudioForConditionalGeneration

from APIs.gpt import generate_gpt_response

@dataclass
class TestArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    model_path: Optional[str] = field(default=None, metadata={"help": "model dir"})
    data_file: Optional[str] = field(default=None, metadata={"help": "test file"})
    audio_dir: Optional[str] = field(default=None, metadata={"help": "audio dir"})
    out_file: Optional[str] = field(default=None, metadata={"help": "output file for test"})
    batch_size: Optional[int] =  field(default=16, metadata={"help": "batch size"})
    force: Optional[bool] = field(default=False, metadata={"help": "force test"})
    mode: Optional[str] = field(default="evaluation", metadata={"help": "mode"})
    temperature: Optional[float] = field(default=0.7, metadata={"help": "temperature"})

    def __post_init__(self):
        if self.model_path is None:
            raise ValueError("config path should not none")


def entropy_aware_decode(model, processor, inputs, top_k=5, temperature=1.0, max_length=128):
    eos_token_id = processor.tokenizer.eos_token_id
    decoder_input_ids = torch.tensor([[processor.tokenizer.bos_token_id]], device=model.device)

    for _ in range(max_length):
        outputs = model(input_features=inputs["input_features"], decoder_input_ids=decoder_input_ids)
        logits = outputs.logits[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)

        top_probs, top_ids = probs.topk(top_k)
        entropies = []

        for i in range(top_k):
            next_token = top_ids[:, i].unsqueeze(0)
            temp_ids = torch.cat([decoder_input_ids, next_token], dim=-1)
            next_logits = model(input_features=inputs["input_features"], decoder_input_ids=temp_ids).logits[:, -1, :]
            next_probs = F.softmax(next_logits, dim=-1)
            entropy = -(next_probs * next_probs.log()).sum().item()
            entropies.append(entropy)

        best_token = top_ids[0, torch.tensor(entropies).argmax()].item()
        decoder_input_ids = torch.cat([decoder_input_ids, torch.tensor([[best_token]], device=model.device)], dim=-1)

        if best_token == eos_token_id:
            break

    return processor.batch_decode(decoder_input_ids[:, 1:], skip_special_tokens=True)[0]


def _get_audio(wav_path):
    waveform, sample_rate = torchaudio.load(wav_path)
    if sample_rate != 16000:
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
    audio = waveform[0]
    return audio


def _get_message(obj_dict, mode):
    if mode == "caption":
        if obj_dict["task"] == "sound":
            question_template = (
                f"Please describe all the events and sounds occurring in the audio clip in detail. Identify "
                f"and describe each sound source, such as objects, animals, weather, or environmental "
                f"noises. Include information about the sequence of events and any interactions between "
                f"sound sources. Mention the context or setting if it can be inferred from the sounds."
            )
        elif obj_dict["task"] == "music":
            question_template = (
                f"Please provide a detailed description of the music in the audio clip. Include information "
                f"about the genre, instruments, tempo, mood, and any notable melodies or harmonies. Describe any "
                f"vocals present, including lyrics if they are clear and discernible. Mention the overall atmosphere "
                f"and emotions conveyed by the music."
            )
        elif obj_dict["task"] == "speech":
            question_template = (
                f"Please transcribe the spoken words in the audio clip accurately. Capture all spoken "
                f"content verbatim, including any significant pauses, emotions, or emphasis expressed by "
                f"the speaker. Do not include interpretations or descriptions beyond the spoken words."
            )
    elif mode == "caption_with_question":
        if obj_dict["task"] == "sound":
            question_template = (
                f"I am asked to answer the question: \"{obj_dict['question']}\". "
                f"Please describe all the events and sounds occurring in the audio clip in detail. Identify "
                f"and describe each sound source, such as objects, animals, weather, or environmental "
                f"noises. Include information about the sequence of events and any interactions between "
                f"sound sources. Mention the context or setting if it can be inferred from the sounds."
            )
        elif obj_dict["task"] == "music":
            question_template = (
                f"I am asked to answer the question: \"{obj_dict['question']}\". "
                f"Please provide a detailed description of the music in the audio clip. Include information "
                f"about the genre, instruments, tempo, mood, and any notable melodies or harmonies. Describe any "
                f"vocals present, including lyrics if they are clear and discernible. Mention the overall atmosphere "
                f"and emotions conveyed by the music."
            )
        elif obj_dict["task"] == "speech":
            question_template = (
                f"I am asked to answer the question: \"{obj_dict['question']}\". "
                f"Please transcribe the spoken words in the audio clip accurately. Capture all spoken "
                f"content verbatim, including any significant pauses, emotions, or emphasis expressed by "
                f"the speaker. Do not include interpretations or descriptions beyond the spoken words."
            )
    else:
        choice_str = f"Please choose the answer from the following options: {obj_dict['choices']}."
        question_template = f"{obj_dict['question']} {choice_str} Output the final answer in <answer> </answer>."
    # If you want to improve the thinking process, uncomment the next line and design your strategy.
    # question_template = f"{obj_dict['question']} {choice_str} Output the thinking process in <think> </think> and final answer in <answer> </answer>."
    print(question_template)

    message = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio_url": f"{obj_dict['audio_id']}"},
                {"type": "text", "text": question_template},
            ],
        }
    ]
    return message

def _get_text_message(obj_dict, mode):
    if mode == "caption":
        choice_str = f"Please choose the answer from the following options: {obj_dict['choices']}."
        question_template = f"{obj_dict['question']} {choice_str} Output the final answer in <answer> </answer>."
    elif mode == "5-caption-evaluation":
        choice_str = f"Please choose the answer from the following options: {obj_dict['choices']}."
        question_template = (
            f"{obj_dict['question'].replace('given audio', '5 audio captions for the same audio')} "
            f"{choice_str} Output the final answer in <answer> </answer>.\n"
            f"Audio caption 1: {obj_dict['model_prediction'][0]}\n"
            f"Audio caption 2: {obj_dict['model_prediction'][1]}\n"
            f"Audio caption 3: {obj_dict['model_prediction'][2]}\n"
            f"Audio caption 4: {obj_dict['model_prediction'][3]}\n"
            f"Audio caption 5: {obj_dict['model_prediction'][4]}\n"
        )
    elif mode == "5-caption-evaluation-cot":
        choice_str = f"Please choose the answer from the following options: {obj_dict['choices']}."
        question_template = (
            f"{obj_dict['question'].replace('given audio', '5 audio captions for the same audio')} "
            f"{choice_str} Please think step by step and output the thinking process in <think> </think> and final answer in <answer> </answer>.\n"
            f"Audio caption 1: {obj_dict['model_prediction'][0]}\n"
            f"Audio caption 2: {obj_dict['model_prediction'][1]}\n"
            f"Audio caption 3: {obj_dict['model_prediction'][2]}\n"
            f"Audio caption 4: {obj_dict['model_prediction'][3]}\n"
            f"Audio caption 5: {obj_dict['model_prediction'][4]}\n"
        )
    elif mode == "1-caption-evaluation":
        choice_str = f"Please choose the answer from the following options: {obj_dict['choices']}."
        question_template = (
            f"{obj_dict['question'].replace('given audio', 'audio caption')} "
            f"{choice_str} Output the final answer in <answer> </answer>.\n"
            f"Audio caption: {obj_dict['model_prediction']}"
        )
    elif mode == "1-caption-evaluation-cot":
        choice_str = f"Please choose the answer from the following options: {obj_dict['choices']}."
        question_template = (
            f"{obj_dict['question'].replace('given audio', 'audio caption')} "
            f"{choice_str} Please think step by step and output the thinking process in <think> </think> and final answer in <answer> </answer>.\n"
            f"Audio caption: {obj_dict['model_prediction']}"
        )

    # If you want to improve the thinking process, uncomment the next line and design your strategy.
    # question_template = f"{obj_dict['question']} {choice_str} Output the thinking process in <think> </think> and final answer in <answer> </answer>."
    print(question_template)

    return question_template

def get_qwen2_audio_response(data_args):
    audio_processor = AutoProcessor.from_pretrained(data_args.model_path)
    audio_model = Qwen2AudioForConditionalGeneration.from_pretrained(data_args.model_path, torch_dtype=torch.bfloat16, device_map="auto")
    audio_model.generation_config.temperature = data_args.temperature


    datas = []

    with open(data_args.data_file, "r") as f:
        datas = json.load(f)
    
    # datas = datas[:10]

    all_outputs = []
    batch_size = data_args.batch_size
    for i in tqdm(range(0, len(datas), batch_size)):
        batch_data = datas[i : i + batch_size]

        batch_messages = []
        batch_audios = []
        for bd in batch_data:
            batch_messages.append(_get_message(bd, data_args.mode))
            audio_path = os.path.join(data_args.audio_dir, bd["audio_id"])
            batch_audios.append(_get_audio(audio_path).numpy())

        text = [
            audio_processor.apply_chat_template(msg, add_generation_prompt=True, tokenize=False)
            for msg in batch_messages
        ]
        inputs = audio_processor(
            text=text, audios=batch_audios, sampling_rate=16000, return_tensors="pt", padding=True
        ).to(audio_model.device)

        generated_ids = audio_model.generate(**inputs, max_new_tokens=1024)

        generated_ids = generated_ids[:, inputs.input_ids.size(1) :]
        response = audio_processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        all_outputs.extend(response)
        print(f"Processed batch {i//batch_size + 1}/{(len(datas) + batch_size - 1)//batch_size}")
    return datas, all_outputs

def get_gpt4o_response(data_args):
    datas = []
    with open(data_args.data_file, "r") as f:
        datas = json.load(f)
    
    all_outputs = []
    batch_size = data_args.batch_size
    for i in tqdm(range(0, len(datas), batch_size)):
        batch_data = datas[i : i + batch_size]


        batch_response = []
        for bd in batch_data:
            response = generate_gpt_response(data_args.model_path, _get_text_message(bd, data_args.mode))
            batch_response.append(response)

        all_outputs.extend(batch_response)
        print(f"Processed batch {i//batch_size + 1}/{(len(datas) + batch_size - 1)//batch_size}")
    return datas, all_outputs



def compute_log_p_y(text):
    ids = gpt2_tokenizer(text, return_tensors="pt").input_ids.to("cuda")
    with torch.no_grad():
        loss = gpt2(input_ids=ids, labels=ids).loss.item()
    return -loss * ids.size(1)

def mi_rerank(model, processor, inputs, num_return_sequences=5):
    outputs = model.generate(
        input_features=inputs["input_features"],
        do_sample=True,
        temperature=1.0,
        num_return_sequences=num_return_sequences,
        max_length=128,
        return_dict_in_generate=True,
        output_scores=True
    )

    decoded = processor.batch_decode(outputs.sequences, skip_special_tokens=True)
    mi_scores = []

    for i, text in enumerate(decoded):
        log_py = compute_log_p_y(text)
        whisper_logp = sum(outputs.scores[j][i, token_id].item()
                           for j, token_id in enumerate(outputs.sequences[i][1:]))
        mi = whisper_logp - log_py
        mi_scores.append((text, mi))

    best_text = sorted(mi_scores, key=lambda x: x[1], reverse=True)[0][0]
    return best_text

def masked_attention_decode(model, processor, inputs, window=0, max_length=128):
    eos_token_id = processor.tokenizer.eos_token_id
    decoder_input_ids = torch.tensor([[processor.tokenizer.bos_token_id]], device=model.device)

    for step in range(max_length):
        # 只保留最近 window 个 token
        truncated_ids = decoder_input_ids[:, -window:] if window > 0 else decoder_input_ids[:, -1:]

        outputs = model(input_features=inputs["input_features"], decoder_input_ids=truncated_ids)
        logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(logits, dim=-1).unsqueeze(-1)
        decoder_input_ids = torch.cat([decoder_input_ids, next_token], dim=-1)

        if next_token.item() == eos_token_id:
            break

    return processor.batch_decode(decoder_input_ids[:, 1:], skip_special_tokens=True)[0]

import torch
import torch.nn.functional as F
from transformers import WhisperProcessor, WhisperForConditionalGeneration

def entropy_aware_decode_robust(model, processor, inputs, top_k=5, temperature=0.8, max_length=128, alpha=0.3):
    eos_token_id = processor.tokenizer.eos_token_id
    bos_token_id = processor.tokenizer.bos_token_id

    # decoder_input_ids = torch.tensor([[bos_token_id]], device=model.device)
    decoder_input_ids = torch.tensor([processor.get_decoder_prompt_ids(language="en", task="transcribe")], device=model.device)

    token_history = []
    repeat_count = {}

    for step in range(max_length):
        # Get logits for current step
        outputs = model(input_features=inputs["input_features"], decoder_input_ids=decoder_input_ids)
        # outputs = model(**inputs, decoder_input_ids=decoder_input_ids)

        logits = outputs.logits[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)

        top_probs, top_ids = probs.topk(top_k)
        entropies = []
        scores = []

        for i in range(top_k):
            candidate_id = top_ids[0, i].item()
            temp_ids = torch.cat([decoder_input_ids, torch.tensor([[candidate_id]], device=model.device)], dim=-1)
            next_logits = model(input_features=inputs["input_features"], decoder_input_ids=temp_ids).logits[:, -1, :]
            next_probs = F.softmax(next_logits, dim=-1)

            entropy = -(next_probs * next_probs.log()).sum().item()
            prob_score = probs[0, candidate_id].item()
            combined_score = 0.0 * entropy + alpha * prob_score  # 融合熵和概率得分

            entropies.append(entropy)
            scores.append(combined_score)

        best_i = torch.tensor(scores).argmax().item()
        best_token = top_ids[0, best_i].item()

        # 重复检测
        if best_token in repeat_count:
            repeat_count[best_token] += 1
        else:
            repeat_count[best_token] = 1

        # if repeat_count[best_token] > 6:
        #     print("⚠️ Too much repetition, stopping decode.")
        #     break

        token_history.append(best_token)
        decoder_input_ids = torch.cat([decoder_input_ids, torch.tensor([[best_token]], device=model.device)], dim=-1)

        # 打印每一步调试日志
        print(f"Step {step}: Token = {processor.tokenizer.decode([best_token])}, Score = {scores[best_i]:.2f}, Entropy = {entropies[best_i]:.2f}")

        if best_token == eos_token_id:
            break

    # Decode final result
    return processor.batch_decode(decoder_input_ids[:, 1:], skip_special_tokens=True)[0]


def get_whisper_response(data_args):


    datas = []
    with open(data_args.data_file, "r") as f:
        datas = json.load(f)
    
    all_outputs = []
    processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
    # model = whisper.load_model("large-v3")



    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3").to("cuda")

    for bd in datas:
        # print(bd)
        if bd["task"] != "speech":
            continue
        audio_path = os.path.join(data_args.audio_dir, bd["audio_id"])
        waveform, sr = torchaudio.load(audio_path)  
        if sr != 16000:
            waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(waveform)
            sr = 16000
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0)  # 将立体声变为单声道
        else:
            waveform = waveform.squeeze(0)
        inputs = processor(waveform, sampling_rate=sr, return_tensors="pt").to("cuda")

        # asr_result = model.transcribe(audio_path, word_timestamps=True, temperature=0.8)["text"]
        text_entropy = entropy_aware_decode_robust(model, processor, inputs, top_k=5, temperature=0.8, alpha=1.0)
        # text_masked = masked_attention_decode(model, processor, inputs, window=1)
        print(text_entropy)
        # print(text_masked)
        assert False
        # all_outputs.append(text_masked)
        # print(text_masked)
        wcn = generate_wcn(model, processor, audio_path)
        all_outputs.append(wcn)
        # print(wcn)
        # assert False

    return datas, all_outputs

def align_sequences(hypotheses):
    """Align multiple ASR hypotheses using Levenshtein alignment."""
    base = hypotheses[0]  # Use first hypothesis as reference
    alignment = [list(base)]  # Initialize alignment grid

    for hyp in hypotheses[1:]:
        matcher = SequenceMatcher(None, base, hyp)
        aligned_hyp = []
        base_idx = 0

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                aligned_hyp.extend(hyp[j1:j2])
            elif tag == "insert":
                aligned_hyp.extend(hyp[j1:j2])
            elif tag == "replace":
                aligned_hyp.extend(hyp[j1:j2])
            elif tag == "delete":
                aligned_hyp.extend(["-"] * (i2 - i1))  # Placeholder for missing words

            base_idx = i2

        alignment.append(aligned_hyp)

    return list(map(list, zip(*alignment)))  # Transpose for word-position alignment


def transcribe_with_timestamps(audio_path, model, t, segment_length=1.0):
    # Get the full transcription with timestamps
    result = model.transcribe(
        audio_path,
        word_timestamps=True,
        temperature=t,  # Enable word-level timestamps
    )
    
    # Initialize containers for segments
    current_segment = []
    current_start = 0
    segments_by_second = []

    # Process each word
    for segment in result["segments"]:
        for word in segment["words"]:
            start_time = word["start"]
            text = word["word"]
            
            # If we've passed the segment length or it's the first word
            if start_time - current_start >= segment_length or not current_segment:
                if current_segment:
                    segments_by_second.append({
                        "start": current_start,
                        "end": start_time,
                        "text": "".join(current_segment).strip()
                    })
                current_segment = [text]
                current_start = start_time
            else:
                current_segment.append(text)

    # Add the last segment if there's anything remaining
    if current_segment:
        segments_by_second.append({
            "start": current_start,
            "end": result["segments"][-1]["end"],
            "text": "".join(current_segment).strip()
        })

    print("segments_by_second:")
    for segment in segments_by_second:
        start_time = str(timedelta(seconds=round(segment["start"])))
        end_time = str(timedelta(seconds=round(segment["end"])))
        print(f"[{start_time} -> {end_time}] {segment['text']}")

    # print(" ".join([s["text"] for s in segments_by_second]))
    # assert False
    return " ".join([s["text"] for s in segments_by_second])


def generate_wcn(model, processor, audio_path):
    hypotheses = [transcribe_with_timestamps(audio_path, model, t).split() for t in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]]
    # hypotheses = [transcribe_with_timestamps(audio_path, model, t).split() for t in [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]

    aligned_cn = align_sequences(hypotheses)
    # print(aligned_cn)
    word_counts = defaultdict(lambda: defaultdict(int))

    for index, position in enumerate(aligned_cn):  # 使用索引
        for word in position:
            if word != "-":  
                word_counts[index][word] += 1  # 用 index 作为键

    # print(word_counts)
    # Convert counts to probabilities
    word_probs = {
        pos: {word: count / sum(words.values()) for word, count in words.items()}
        for pos, words in word_counts.items()
    }
    # print(word_probs)

    # Print the Word Confusion Network
    print("\nWord Confusion Network (WCN):")
    for pos, words in word_probs.items():
        print(f"Position {pos}: {', '.join([f'{w} ({p:.2f})' for w, p in words.items()])}")

    # Print the WCN in a string format
    wcn_str = ""
    for pos, words in word_probs.items():
        wcn_str += f"Position {pos}: {', '.join([f'{w} ({p:.2f})' for w, p in words.items()])}"
    return wcn_str

def main():
    parser = HfArgumentParser(TestArguments)
    data_args = parser.parse_args_into_dataclasses()[0]
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    transformers.logging.set_verbosity_info()
    logging.info(data_args)

    if not data_args.force and os.path.exists(data_args.out_file) and os.path.getsize(data_args.out_file) > 0:
        logging.info(f"The {data_args.out_file} exists. Do not regenerate it.")
        return
    
    out_dir = os.path.abspath(os.path.dirname(data_args.out_file))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if "gpt" in data_args.model_path:
        datas, all_outputs = get_gpt4o_response(data_args)
    elif "whisper" in data_args.model_path:
        datas, all_outputs = get_whisper_response(data_args)
    else:
        datas, all_outputs = get_qwen2_audio_response(data_args)

    def extract_answer(output_str):
        answer_pattern = r"<answer>(.*?)</answer>"
        match = re.search(answer_pattern, output_str)

        if match:
            return match.group(1)
        return output_str
    
    def extract_cot(output_str):
        cot_pattern = r"<think>(.*?)</think>"
        match = re.search(cot_pattern, output_str)
        if match:
            return match.group(1)
        return ""

    final_output = []
    for input_example, model_output in zip(datas, all_outputs):
        original_output = model_output

        model_answer = extract_answer(original_output).strip()
        if "cot" in data_args.mode:
            model_cot = extract_cot(original_output).strip()

        # Create a result dictionary for this example
        result = input_example
        result[f"model_prediction"] = model_answer
        if "cot" in data_args.mode:
            result[f"model_cot"] = model_cot

        final_output.append(result)

    
    # Save results to a JSON file
    output_path = data_args.out_file
    with open(output_path, "w") as f:
        json.dump(final_output, f, indent=2)

    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
