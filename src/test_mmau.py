import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Optional
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
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
    mode: Optional[str] = field(default="inference", metadata={"help": "mode"})
    temperature: Optional[float] = field(default=0.7, metadata={"help": "temperature"})

    def __post_init__(self):
        if self.model_path is None:
            raise ValueError("config path should not none")


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
    elif mode == "1-caption-evaluation":
        choice_str = f"Please choose the answer from the following options: {obj_dict['choices']}."
        question_template = (
            f"{obj_dict['question'].replace('given audio', 'audio caption')} "
            f"{choice_str} Output the final answer in <answer> </answer>.\n"
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
    else:
        datas, all_outputs = get_qwen2_audio_response(data_args)

    def extract_answer(output_str):
        answer_pattern = r"<answer>(.*?)</answer>"
        match = re.search(answer_pattern, output_str)

        if match:
            return match.group(1)
        return output_str

    final_output = []
    for input_example, model_output in zip(datas, all_outputs):
        original_output = model_output
        model_answer = extract_answer(original_output).strip()

        # Create a result dictionary for this example
        result = input_example
        result[f"model_prediction"] = model_answer
        final_output.append(result)

    
    # Save results to a JSON file
    output_path = data_args.out_file
    with open(output_path, "w") as f:
        json.dump(final_output, f, indent=2)

    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
