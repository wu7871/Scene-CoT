import sys
from rllm.system_prompts import SCENECOT_SYSTEM_PROMPT
import os
import json
import argparse
import time
import re
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
import pandas as pd

from transformers import AutoModelForCausalLM, AutoTokenizer

# os.environ["CUDA_VISIBLE_DEVICES"] = "7"

def init_model(model_name):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="cuda",
        torch_dtype="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def read_files_and_tokenize(parquet_file):
    dataframe = pd.read_parquet(parquet_file)
    print(f'original dataset len: {len(dataframe)}')
    return dataframe

OUTPUT_PATH = "./results"


BATCH_SIZE = 8

def generate_batch(chats, model, tokenizer):
    prompt_chats = [
        [
            {'role': 'system', 'content': SCENECOT_SYSTEM_PROMPT},
            {'role': 'user', 'content': chat}
        ] for chat in chats
    ]
    prompt_with_chat_templates = [
        tokenizer.apply_chat_template(pc, add_generation_prompt=True, tokenize=False)
        for pc in prompt_chats
    ]
    model_inputs = tokenizer(prompt_with_chat_templates, return_tensors="pt", padding=True).to(model.device)
    generated_ids = model.generate(
        **model_inputs,
        temperature=0.7,
        max_new_tokens=24576
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return responses

def inference(model_name, prompt_key='prompt'):
    if 'actor' in model_name:
        dir_name = model_name.split('/')[-3] + '_' + model_name.split('/')[-1]
    else:
        dir_name = model_name.split('/')[-1]
    save_path = os.path.join(OUTPUT_PATH, dir_name)
    os.makedirs(save_path, exist_ok=True)
    model, tokenizer = init_model(model_name)
    dataframe = read_files_and_tokenize("./rllm/data/test_scenecot.parquet")
    total = 0
    format_correct = 0

    for start in range(0, len(dataframe), BATCH_SIZE):
        batch_rows = dataframe.iloc[start:start+BATCH_SIZE]
        chats = []
        filenames = []
        filenames_think = []
        for item in range(len(batch_rows)):
            row_dict = batch_rows.iloc[item].to_dict()
            chat = row_dict.pop(prompt_key).item()['content']
            user_preference = chat.split("You FIRST think about")[0].strip()
            filename = re.sub(r'[^a-zA-Z0-9]', '_', user_preference) + '.json'
            filename_think = re.sub(r'[^a-zA-Z0-9]', '_', user_preference) + '_think.json'
            chat = "User Preferences: " + chat + "Importantly, for relative placement with other objects in the room use the prepositions 'on', 'left of', 'right of', 'in front', 'behind', 'under'. For relative placement with the room layout elements (walls, the middle of the room, ceiling) use the prepositions 'on', 'in the corner'. You are not allowed to use any prepositions different from the ones above!!"
            chats.append(chat)
            filenames.append(filename)
            filenames_think.append(filename_think)
        generated_responses = generate_batch(chats, model, tokenizer)

        for i, generated_response in enumerate(generated_responses):
            total += 1
            save_file_path = os.path.join(save_path, filenames[i])
            save_file_path_think = os.path.join(save_path, filenames_think[i])
            if '<answer>' in generated_response and '</answer>' in generated_response:
                start_idx = generated_response.find('<answer>') + len('<answer>')
                end_idx = generated_response.find('</answer>')
                answer_content = generated_response[start_idx:end_idx].strip()
                try:
                    json.loads(answer_content)
                    format_correct += 1
                    answer = answer_content
                    with open(save_file_path, 'w', encoding='utf-8') as f:
                        f.write(answer)
                    print(f"Save to {save_file_path}")
                except Exception:
                    pass
            if '<think>' in generated_response and '</think>' in generated_response:
                start_idx = generated_response.find('<think>') + len('<think>')
                end_idx = generated_response.find('</think>')
                answer_content = generated_response[start_idx:end_idx].strip()
                try:
                    json.loads(answer_content)
                    answer = answer_content
                    with open(save_file_path_think, 'w', encoding='utf-8') as f:
                        f.write(answer)
                    print(f"Save to {save_file_path_think}")
                except Exception:
                    pass

    if total != 0:
        with open(os.path.join(save_path, 'format_accuracy.txt'), 'w', encoding='utf-8') as f:
            f.write(f"Format Accuracy: {format_correct / total}")
    return

MODEL_LIST = [
    "./model",
]
if __name__ == "__main__":
    
    for model in MODEL_LIST:
        inference(model)

