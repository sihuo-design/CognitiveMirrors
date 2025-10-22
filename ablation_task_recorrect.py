import sys
import os
import argparse
from tqdm import tqdm
import logging
import os
import time
import torch
import gc
import json
import copy
import pandas as pd
from collections import defaultdict
import pickle

import numpy as np
import random
from utils import cot_prompt, kl_scoring, simi_scoring, get_intervation_result, emb_similarity
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from interveners import wrapper, Collector, ITI_Intervener, head_Intervener
import pyvene as pv
from sentence_transformers import SentenceTransformer
import csv
from datasets import load_dataset
import re
from llm import efficient_openai_text_api
from einops import rearrange
import evaluate
from collections import Counter
from tqdm import tqdm

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PL_NO_SLURM"] = "1"
os.environ["SLURM_JOB_ID"] = ""
os.environ["SLURM_ARRAY_TASK_ID"] = ""

torch.set_float32_matmul_precision('high')

print("[环境设置] TOKENIZERS_PARALLELISM=false, matmul_precision=high 已设置")

HF_NAMES = {
    # 'llama_7B': 'baffo32/decapoda-research-llama-7B-hf',
    'llama_7B': 'huggyllama/llama-7b',
    'alpaca_7B': 'circulus/alpaca-7b', 
    'vicuna_7B': 'AlekseyKorshuk/vicuna-7b', 
    'llama2_chat_7B': 'meta-llama/Llama-2-7b-chat-hf', 
    'llama2_chat_13B': 'meta-llama/Llama-2-13b-chat-hf', 
    'llama2_chat_70B': 'meta-llama/Llama-2-70b-chat-hf', 
    'llama3_8B': 'meta-llama/Meta-Llama-3-8B',
    'llama3_8B_deepseek': 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B',
    'llama3_8B_instruct': 'meta-llama/Meta-Llama-3-8B-Instruct',
    'llama3_70B': 'meta-llama/Meta-Llama-3-70B',
    'llama3_70B_instruct': 'meta-llama/Meta-Llama-3-70B-Instruct',
    'llama3.1_8B_instruct': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
    'llama3.2_3B_instruct': 'meta-llama/Llama-3.2-3B-Instruct',
    'llama3_70B': 'meta-llama/Meta-Llama-3-70B',
    'llama3_70B_instruct': 'meta-llama/Meta-Llama-3-70B-Instruct',
    'qwen3-8B': 'Qwen/Qwen3-8B',
    'qwen3-1.7B': 'Qwen/Qwen3-1.7B',
    'qwen3-4B': 'Qwen/Qwen3-4B',
    'gemma3-12B': 'google/gemma-3-12b-it',
    'gemma3-4B': 'google/gemma-3-4b-it',
    'phi4-3.8B': "microsoft/Phi-4-mini-instruct",
    'internlm2-1.8B': 'internlm/internlm2-1_8b',
    'yi-1.5-6B': '01-ai/Yi-1.5-6B-Chat',
    'yi-1.5-9B': '01-ai/Yi-1.5-9B-Chat'
}


def flattened_idx_to_layer_head(flattened_idx, num_heads):
    return flattened_idx // num_heads, flattened_idx % num_heads

def layer_head_to_flattened_idx(layer, head, num_heads):
    return layer * num_heads + head

def run_intervention(inter_heads_idx, com_directions):

    pv_config = []
    for layer_idx, heads in inter_heads_idx.items():
        direction = torch.zeros(head_dim * num_heads).to("cpu")
        for head in heads:
            dir = torch.tensor(com_directions[layer_head_to_flattened_idx(layer_idx, head, num_heads)], dtype=torch.float32).to("cpu")
            dir = dir / torch.norm(dir)
            activations = torch.tensor(tuning_activations[:,layer_idx,head,:], dtype=torch.float32).to("cpu") # batch x 128
            proj_vals = activations @ dir.T
            proj_val_std = torch.std(proj_vals)
            direction[head * head_dim: (head + 1) * head_dim] = dir * proj_val_std
        intervener = ITI_Intervener(direction, args.alpha) #head=-1 to collect all head activations, multiplier doens't matter
        # interveners.append(intervener)
        intervener.layer_idx = layer_idx 
        pv_config.append({
            "component": f"model.layers[{layer_idx}].self_attn.o_proj.input",
            "intervention": wrapper(intervener),
        })
    intervened_model = pv.IntervenableModel(pv_config, model)
    
    return intervened_model

def run_intervention_all(inter_heads_idx, inter_heads_functions, sorted_stats, functions_com_directions):

    pv_config = []
    for layer_idx, heads in inter_heads_idx.items():
        direction = torch.zeros(head_dim * num_heads).to("cpu")
        for head in heads:
            # dir = torch.tensor(com_directions[layer_head_to_flattened_idx(layer_idx, head, num_heads)], dtype=torch.float32).to("cpu")
            idx = layer_head_to_flattened_idx(layer_idx, head, num_heads)
            functions = inter_heads_functions[idx]
            dir = torch.zeros(head_dim).to("cpu")
            for function in functions:
                dir = dir + torch.tensor(functions_com_directions[function][idx], dtype=torch.float32).to("cpu")
            dir = dir / len(functions)
            dir = dir / torch.norm(dir)
            activations = torch.tensor(tuning_activations[:,layer_idx,head,:], dtype=torch.float32).to("cpu") # batch x 128
            proj_vals = activations @ dir.T
            proj_val_std = torch.std(proj_vals)
            direction[head * head_dim: (head + 1) * head_dim] = dir * proj_val_std
        intervener = ITI_Intervener(direction, args.alpha) #head=-1 to collect all head activations, multiplier doens't matter
        # interveners.append(intervener)
        intervener.layer_idx = layer_idx  # 注意：为了让 __call__ 能访问当前层索引 
        pv_config.append({
            "component": f"model.layers[{layer_idx}].self_attn.o_proj.input",
            "intervention": wrapper(intervener),
        })
    intervened_model = pv.IntervenableModel(pv_config, model)
    
    return intervened_model

def get_intervention_heads(layer_idx, head_idx, num_layers, num_heads):
    """
    getting the intervention heads index
    """
    return {layer_idx: [head_idx]}

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    try:
        import transformers
        transformers.set_seed(seed)
    except ImportError:
        pass

def get_topk_intervention_heads(head_features):
    """
    getting the topk intervention heads index
    :param head_features: List[int]
    :param num_heads: int, the number of heads per layer
    :return: dict, layer_idx -> List[head_idx]
    """
    inter_heads_idx = {}
    for feature in head_features:
        if type(feature) != int:
            feature = feature[0]
        layer_idx = feature // num_heads
        head_idx = feature % num_heads
        if layer_idx not in inter_heads_idx:
            inter_heads_idx[layer_idx] = [head_idx]
        else:
            inter_heads_idx[layer_idx].append(head_idx)
    
    return inter_heads_idx

def get_last_intervention_heads(head_features):
    save_heads_idx = {}
    inter_heads_idx = {}
    for feature in head_features:
        if type(feature) != int:
            feature = feature[0]
        layer_idx = feature // num_heads
        head_idx = feature % num_heads
        if layer_idx not in save_heads_idx:
            save_heads_idx[layer_idx] = [head_idx]
        else:
            save_heads_idx[layer_idx].append(head_idx)

    for i in range(num_layers):
        inter_heads_idx[i] = list(range(num_heads))
        if i in save_heads_idx:
            for head in save_heads_idx[i]:
                inter_heads_idx[i].remove(head)
    return inter_heads_idx

def convert(obj):
    if isinstance(obj, dict):
        return {k: convert(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert(i) for i in obj]
    elif isinstance(obj, np.float32) or isinstance(obj, np.float64):
        return float(obj)
    else:
        return obj
    
system_role = """
You are an expert in analytical and logical reasoning. Your task is to answer the question.
"""

prompt = """
{system_role}
Here is the question:
<question>
{question}
</question>

Instructions:
    - Think through the problem step by step.
    - Provide the final answer along with a brief explanation.
    - Be concise and avoid unnecessary details.

Only output your final answer using this format:
[
  {{
    "answer": "<Your final answer here>"
    "explanation": "<Your explanation here>"
  }}
]

Your answer:
"""

llama_prompt = """
{system_role}
Here is the question:
<question>
{question}
</question>

Instructions:
- Think through the problem carefully and logically.
- Provide a step-by-step explanation leading to the final answer.
- Be concise and avoid unnecessary information.    

Output format:
[
  {{
    "explanation": "<Your explanation here>",
    "final answer": "<Your final answer here>"
  }}
]

Your answer:
"""    

other_prompt = """
{system_role}
Here is the question:
<question>
{question}
</question>

Instructions:
- Think through the problem step by step to provide the explanation and the final answer.
- Be concise and avoid unnecessary information.   
- Your output must strictly follow the format shown below. 

Output format:
[
  {{
    "explanation": "<Your step-by-step thinking process here>",
    "final answer": "<Your final answer here>"
  }}
]

Your answer:
"""

extraction_prompt = """

You are a document information assistant. Please answer the user's question using only exact sentences copied from the document content. Do not paraphrase or infer beyond what is explicitly stated.

<Document Content>
{document_text}
</Document Content>

<Question>
{question}
</Question>

Extract the most relevant sentence(s) from the document that directly answer the question.

Output format:
[
  {{
    "answer": "<answer here>"
  }}
]
"""

gpt_prompt_onlyanswer = """
You are given a reference answer and a predicted answer to a question.

This is the reference answer:
<reference_answer>
{reference_answer}
</reference_answer>

This is the predicted answer:
<prediction_answer>
{prediction_answer}
</prediction_answer>

Instructions:
1. Return `"True"` if the prediction answer is consistent to the reference answer. Otherwise, return `"False"`.
2. Provide a **confidence score** between `0` and `1` based on how certain you are.

Output format:
[
{{
  "correct": <True|False>,
  "confidence": <float between 0 and 1>
}}
]
Your answer:
"""

def get_output(prompt, model, tokenizer, device):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=512, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_only_ids = outputs[0][inputs["input_ids"].shape[1]:].tolist()
    output_answer = tokenizer.decode(generated_only_ids, skip_special_tokens=True)
    answer = ''
    explanation = ''
    match = re.search(r'"answer"\s*:\s*["\']?(.*?)["\']?,', output_answer)
    if match:
        answer = match.group(1)
        print("Answer:", answer)
    match = re.search(r'"explanation"\s*:\s*"(.+?)"\s*}', output_answer, re.DOTALL)
    if match:
        explanation = match.group(1).strip()
        print("Explanation:", explanation)
    else:
        print("No explanation found.")
    return answer, explanation

def input_prompt(dataset):
    all_prompts = []
    
    k = 0
    for i in range(len(dataset)):
        ref_answer = dataset[i]['answer']
        pre_answer = dataset[i]['response']["answer"]

        # format prompt
        prompt_text = gpt_prompt_onlyanswer.format(
            reference_answer=ref_answer,
            prediction_answer=pre_answer
        )
        k += 1

        all_prompts.append(prompt_text)

    return all_prompts

def get_result_from_output(outputs):
    answer_list = []
    confidence_list = []
    
    for i, output in enumerate(outputs):
        output = output[0][0].replace("True", "true").replace("False", "false")
        matches = re.findall(r'\[\s*{[^}]+}\s*\]', output)
        if matches:
            output_json_block = matches[-1].strip()
            print("Extracted JSON answer block:", output_json_block)

            try:
                # parsed = safe_json_parse(output_json_block)
                parsed = json.loads(output_json_block)
                answer, confidence = parsed[0]["correct"], parsed[0]["confidence"]  # This will return '"$80"'
                print("Final extracted answer (with quotes):", answer)
            except:
                answer_match = re.search(r'"correct":\s*"([^"]+)"', output)
                confidence_match = re.search(r'"confidence":\s*([\d.]+)', output)
                
                if answer_match and confidence_match:
                    answer = answer_match.group(1)
                    confidence = float(confidence_match.group(1))

                else:
                    answer = ""
                    confidence = 0
        
        answer_list.append(answer)
        confidence_list.append(confidence)
        
        print(f"Answer: {answer}, Confidence: {confidence}")
    
    return answer_list, confidence_list

def load_data(model_name, layer_num, heads_num, dim, position, head_num=-1, mode="test"):
    if model_name == "qwen3-8B" and mode == "train":
        with open("head_results/qwen3-8B_cot_qa_head_wise_train_0.pkl", 'rb') as f:
            head_results_0 = pickle.load(f) 
        with open("head_results/qwen3-8B_cot_qa_head_wise_train_1.pkl", 'rb') as f:
            head_results_1 = pickle.load(f)
        with open("head_results/qwen3-8B_cot_qa_head_wise_train_2.pkl", 'rb') as f:
            head_results_2 = pickle.load(f)
        with open("head_results/qwen3-8B_cot_qa_head_wise_train_3.pkl", 'rb') as f:
            head_results_3 = pickle.load(f)
        data = head_results_0 + head_results_1 + head_results_2 + head_results_3
    else:
        with open(f'head_results/{model_name}_cot_qa_head_wise_{mode}.pkl', 'rb') as f:
            data = pickle.load(f)
    with open(f'head_results/{model_name}_cot_qa_token_positions_{mode}.pkl', "rb") as f:
        topk_position = pickle.load(f) 

    features = []
    for i in range(len(data)):
        token_num = data[i].shape[1]
        if args.token_use == "topk":
            token_select = [x for x in topk_position[i] if 0 <= x < token_num]

        reshaped = data[i].reshape(layer_num, token_num, heads_num, dim)
        reshaped = reshaped[:, token_select, :, :]

        if position in ["topk", "full"]:
            reshaped = np.mean(reshaped, axis=1)


        reshaped = reshaped.reshape(layer_num * heads_num, dim)

        features.append(reshaped if head_num == -1 else reshaped[head_num])
    return features

def load_data_fine(model_name, layer_num, heads_num, dim, position, head_num=-1):
    with open(f'inter_results/{model_name}/{model_name}_extra_300_head_wise_test.pkl', 'rb') as f:
        data = pickle.load(f)
    with open(f'inter_results/{model_name}/{model_name}_extra_300_token_positions_test.pkl', "rb") as f:
        topk_position = pickle.load(f) 
    
    features = []
    for i in range(len(data)):
        token_num = data[i].shape[1]
        if args.token_use == "topk":
            token_select = [x for x in topk_position[i] if 0 <= x < token_num]

        reshaped = data[i].reshape(layer_num, token_num, heads_num, dim)
        reshaped = reshaped[:, token_select, :, :]

        if position in ["topk", "full"]:
            reshaped = np.mean(reshaped, axis=1)


        reshaped = reshaped.reshape(layer_num * heads_num, dim)

        features.append(reshaped if head_num == -1 else reshaped[head_num])
    return features

def get_com_directions(num_layers, num_heads, usable_idxs, head_wise_activations, answers): 
    labels = []
    for i in usable_idxs:
        labels.append(answers[i][0]['label'])
    str_indices = [i for i, v in enumerate(labels) if isinstance(v, str)]
    for i in str_indices:
        if labels[i] == "True" or labels[i] == "true":
            labels[i] = True
        elif labels[i] == "False" or labels[i] == "false":
            labels[i] = False

    usable_labels = [int(l) for l in labels]
    
    com_directions = []
    # head_wise_activations = head_wise_activations.reshape(num_layers, num_heads, -1)
    for layer in tqdm(range(num_layers), desc="get_com_directions"): 
        for head in range(num_heads): 
            usable_head_wise_activations = [head_wise_activations[i].reshape(num_layers, num_heads, -1)[layer,head,:] for i in usable_idxs]
            usable_head_wise_activations = np.stack(usable_head_wise_activations)
            labels_1 = np.where(np.array(usable_labels) == 1)[0]
            labels_0 = np.where(np.array(usable_labels) == 0)[0]
            true_mass_mean = np.mean(usable_head_wise_activations[labels_1], axis=0)
            false_mass_mean = np.mean(usable_head_wise_activations[labels_0], axis=0)
            com_directions.append(true_mass_mean - false_mass_mean)
    com_directions = np.array(com_directions)

    return com_directions

def get_functions_com_directions(num_layers, num_heads, train_idxs_functions, test_idxs_functions, train_head_wise_activations_all, test_head_wise_activations_all):
    functions_com_directions = {}
    for i in functions_names:
        train_idxs = train_idxs_functions[i]["idx"]
        test_idxs = test_idxs_functions[i]["idx"]
        train_labels = train_idxs_functions[i]["labels"]
        test_labels = test_idxs_functions[i]["labels"]
        if len(train_idxs) == 0 or len(test_idxs) == 0:
            continue
        train_head_wise_activations = [train_head_wise_activations_all[j].reshape(num_layers, num_heads, -1) for j in train_idxs]
        test_head_wise_activations = [test_head_wise_activations_all[j].reshape(num_layers, num_heads, -1) for j in test_idxs]
        functions_com_direction = []
        for layer in tqdm(range(num_layers), desc="get_functions_com_directions"): 
            for head in range(num_heads): 
                usable_train_head_wise_activations = [train_head_wise_activations[j][layer,head,:] for j in range(len(train_head_wise_activations))]
                usable_train_head_wise_activations = np.stack(usable_train_head_wise_activations)
                labels_1 = np.where(np.array(train_labels) == 1)[0]
                labels_0 = np.where(np.array(train_labels) == 0)[0]
                usable_test_head_wise_activations = [test_head_wise_activations[j][layer,head,:] for j in range(len(test_head_wise_activations))]
                usable_test_head_wise_activations = np.stack(usable_test_head_wise_activations)
                labels_1_test = np.where(np.array(test_labels) == 1)[0]
                labels_0_test = np.where(np.array(test_labels) == 0)[0]
                true_usable_head_wise_activations = np.concatenate((usable_train_head_wise_activations[labels_1], usable_test_head_wise_activations[labels_1_test]), axis=0)
                false_usable_head_wise_activations = np.concatenate((usable_train_head_wise_activations[labels_0], usable_test_head_wise_activations[labels_0_test]), axis=0)
                # true_mass_mean = np.mean(usable_train_head_wise_activations[labels_1], axis=0)
                # false_mass_mean = np.mean(usable_train_head_wise_activations[labels_0], axis=0)
                true_mass_mean = np.mean(true_usable_head_wise_activations, axis=0)
                false_mass_mean = np.mean(false_usable_head_wise_activations, axis=0)
                com_direction = true_mass_mean - false_mass_mean
                functions_com_direction.append(com_direction)
        functions_com_direction = np.array(functions_com_direction)
        functions_com_directions[i] = functions_com_direction
    return functions_com_directions


def scoring(base_prediction, reference_answer):
    predictions = [str(base_prediction)]
    references = [str(reference_answer)]
    results = {}

    # BLEU

    bleu_score = bleu.compute(predictions=predictions, references=references)
    results["bleu"] = bleu_score["bleu"]

    # ROUGE
    rouge_score = rouge.compute(predictions=predictions, references=references)
    results.update(rouge_score)  # rouge1, rouge2, rougeL, rougeLsum
    
    cosine_score = emb_similarity([predictions, references])
    results["cosine"] = cosine_score

    # COMET
    if args.use_comet:
        comet_score = comet.compute(
            predictions=predictions,
            references=references,
            sources=references
        )
        results["comet"] = comet_score["mean_score"]
    else:
        results["comet"] = "Skipped (no sources provided)"
    return results

def get_idxs_of_functions(dataset_name, model_name, functions_names):
    if dataset_name == "cot_qa":
        train_path = '/data/projects/punim1970/nips2025/LLM_reasoning/dataset/balanced_cot_train_data.json'
        train_data = json.load(open(train_path, "r"))
        test_path = '/data/projects/punim1970/nips2025/LLM_reasoning/dataset/balanced_cot_test_data.json'
        test_data = json.load(open(test_path, "r")) 
        train_label_path = "head_results/output_" + model_name + "_" + dataset_name + "_train" + "_with_gpt_label.json"
        train_data_label = json.load(open(train_label_path, "r"))
        test_label_path = "head_results/output_" + model_name + "_" + dataset_name + "_test" + "_with_gpt_label.json"
        test_data_label = json.load(open(test_label_path, "r"))
    train_functions = []
    for i in range(len(train_data)):
        generated = train_data[i][0]["generated"]
        for j in range(len(generated)):
            label = generated[j]["cognitive_skill"]
            train_functions.append(label)
    test_functions = []
    for i in range(len(test_data)):
        generated = test_data[i][0]["generated"]
        for j in range(len(generated)):
            label = generated[j]["cognitive_skill"]
            test_functions.append(label)
    train_labels = []
    for i in range(len(train_data_label)):
        train_labels.append(train_data_label[i][0]['label'])
    str_indices = [i for i, v in enumerate(train_labels) if isinstance(v, str)]
    for i in str_indices:
        if train_labels[i] == "True" or train_labels[i] == "true":
            train_labels[i] = True
        elif train_labels[i] == "False" or train_labels[i] == "false" or train_labels[i] == "":
            train_labels[i] = False
    train_labels = [int(l) for l in train_labels]
    
    test_labels = []
    for i in range(len(test_data_label)):
        test_labels.append(test_data_label[i][0]['label'])
    str_indices = [i for i, v in enumerate(test_labels) if isinstance(v, str)]
    for i in str_indices:
        if test_labels[i] == "True" or test_labels[i] == "true":
            test_labels[i] = True
        elif test_labels[i] == "False" or test_labels[i] == "false" or test_labels[i] == "":
            test_labels[i] = False
    test_labels = [int(l) for l in test_labels]
    train_function_list = defaultdict(list)
    test_function_list = defaultdict(list)
    for idx, category in enumerate(train_functions):
        train_function_list[category].append(idx)
    for idx, category in enumerate(test_functions):
        test_function_list[category].append(idx)

    train_idxs_functions = {}
    for i in range(len(functions_names)):
        if functions_names[i] in train_function_list:
            idx = train_function_list[functions_names[i]]
            labels = [train_labels[j] for j in idx]
            save_idx = {
            "idx": idx,
            "labels": labels
            }
            train_idxs_functions[functions_names[i]] = save_idx
        else:
            train_idxs_functions.append([])
    test_idxs_functions = {}
    for i in range(len(functions_names)):
        if functions_names[i] in test_function_list:
            idx = test_function_list[functions_names[i]]
            labels = [test_labels[j] for j in idx]
            save_idx ={"idx": idx, "labels": labels}
            test_idxs_functions[functions_names[i]] = save_idx
        else:
            test_idxs_functions.append([])
    
    return train_idxs_functions, test_idxs_functions


def extract_answer(output_text):
    match = re.search(r"####\s*(-?[0-9,\.]+)", output_text)
    if match:
        return match.group(1).replace(",", "")
    return None

def extract_number(text):
    match = re.search(r'-?\d+(\.\d+)?', text)
    return float(match.group()) if match else None

def extract_explanations_and_answers(text):
    json_blocks = re.findall(r'\[\s*\{.*?\}\s*\]', text, flags=re.DOTALL)
    
    results = []
    for block in json_blocks:
        try:
            data = json.loads(block)
            if isinstance(data, list) and isinstance(data[0], dict):
                explanation = str(data[0].get("explanation", "")).strip()
                final_answer = str(data[0].get("final answer", "")).strip()
                results.append({
                    "explanation": explanation,
                    "final_answer": final_answer
                })
        except Exception as e:
            print(f"Error: {e}")
            continue

    return results

functions_names = ["Retrieval", "Knowledge Recall", "Semantic Understanding", "Syntactic Understanding", "Math Calculation", "Inference", "Logical Reasoning", "Decision-making"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='yi-1.5-6B')
    parser.add_argument('--model_prefix', type=str, default='', help='prefix of model name')
    parser.add_argument('--dataset_name', type=str, default='cot_qa')
    parser.add_argument('--dataset_fine', type=str, default='GSM8K')
    parser.add_argument('--activations_dataset', type=str, default=None)
    parser.add_argument('--intervation_way', type=str, default='single')
    parser.add_argument('--recorrect_all', default=False, action='store_true')
    parser.add_argument('--alpha', type=float, default=5, help='alpha, intervention strength')
    parser.add_argument('--token_use', type=str, default="topk")
    parser.add_argument('--mask_ratio', type=int, default=30)
    parser.add_argument('--mask_ratio_all', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--llm_model_method', default = 'o4-mini', type = str)
    parser.add_argument('--key', default = 'your key', type = str)
    parser.add_argument('--temperature', type=float, default=1)
    parser.add_argument('--function_name', type=str, default='Knowledge Recall')
    parser.add_argument('--use_bleu', default=True, action='store_true')
    parser.add_argument('--use_rouge', default=True, action='store_true')
    parser.add_argument('--use_cosine', default=True, action='store_true')
    parser.add_argument('--use_comet', default=True, action='store_true')
    parser.add_argument('--use_kl', default=False, action='store_true')
    parser.add_argument('--use_head', type=str, default="topk")
    parser.add_argument('--use_layer_bias', default=True, action="store_true")
    parser.add_argument('--output_dir', type=str, default="recorrect_head_acc.csv")
    parser.add_argument('--mode', type=str, default="test")
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    train_idxs_functions, test_idxs_functions = get_idxs_of_functions(args.dataset_name, args.model_name, functions_names)
    with open(f"main_results/{args.model_name}/layer_True_position_topk_elbow.json", "r") as f:
        elbow_index = json.load(f)
    args.mask_ratio = int(elbow_index[args.function_name])

    model_name_or_path = HF_NAMES[args.model_prefix + args.model_name]

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, token='your token', trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, low_cpu_mem_usage=True, token='your token', torch_dtype=torch.float16, device_map="cuda:0")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    device = "cuda"
    # define number of layers and heads
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    hidden_size = model.config.hidden_size
    if args.model_name == "qwen3-4B":
        head_dim = model.config.head_dim
    else:
        head_dim = hidden_size // num_heads    
        
    mask_config = {
        "scale_factor": 0.0001,
        "mask_type": "scale_mask",
    }
    
    
    if args.dataset_fine == "GSM8K":
        dataset_fine = load_dataset("gsm8k", "main") 
        dataset_fine = dataset_fine['test'][:100]        
    elif args.dataset_fine == "extraction_qa":
        path = 'dataset/extraction_qa.json'
        dataset_fine = json.load(open(path, "r"))            

    with open(f"main_results/{args.model_name}/layer_{args.use_layer_bias}_position_{args.token_use}.json", "r") as f:
        importances = json.load(f)
     
    if args.intervation_way == "single":
        if args.use_head == "randomk":
            head_features = importances[args.function_name][:args.mask_ratio]
            all_heads = list(range(num_layers * num_heads))
            all_heads = [h for h in all_heads if h not in head_features]
            head_features = random.sample(all_heads, args.mask_ratio)
        elif args.use_head == "lowk":
            head_features = importances[args.function_name][-args.mask_ratio:]
        elif args.use_head == "topk":
            head_features = importances[args.function_name][:args.mask_ratio]
        
        if args.use_head in ["topk", "randomk", "lowk"]:
            inter_heads_idx = get_topk_intervention_heads(head_features)  
        elif args.use_head == "last":
            inter_heads_idx = get_last_intervention_heads(head_features)
            
    activations_dataset = args.dataset_fine if args.activations_dataset is None else args.activations_dataset    
    tuning_activations = load_data_fine(args.model_name, num_layers, num_heads, head_dim, args.token_use)
    tuning_activations = np.stack(tuning_activations)
    tuning_activations = rearrange(tuning_activations, 'b (l h) d -> b l h d', l=num_layers, h = num_heads)
    
    train_head_wise_activations = load_data(args.model_name, num_layers, num_heads, head_dim, args.token_use, mode="train")
    test_head_wise_activations = load_data(args.model_name, num_layers, num_heads, head_dim, args.token_use, mode="test")
    functions_com_direction = get_functions_com_directions(num_layers, num_heads, train_idxs_functions, test_idxs_functions, train_head_wise_activations, test_head_wise_activations)
    

    if args.intervation_way == "single":
        com_direction = functions_com_direction[args.function_name]
        intervened_model = run_intervention(inter_heads_idx, com_direction)
    

    scores_list = []
    generated_answers = []
    acc_head = {}
    acc_num = 0
    
    if not os.path.exists(os.path.join(f"rebuttal_results/{args.model_name}", args.function_name.replace(" ", "_"))):
        os.makedirs(os.path.join(f"rebuttal_results/{args.model_name}", args.function_name.replace(" ", "_")))

    if args.recorrect_all or args.dataset_fine in ["GSM8K"]:
        dataset_idx = range(len(dataset_fine["question"]))
    elif args.dataset_fine == "extraction_qa":
        dataset_idx = range(len(dataset_fine))
    else:
        labels = []
        with open(f"inter_results/{args.model_name}/output_" + args.model_name + "_" + args.dataset_fine + "_train_with_gpt_label.json", "r") as f:
            answers = json.load(f)
        for i in range(len(answers)):
            labels.append(answers[i][0]['answer2'])
        str_indices = [i for i, v in enumerate(labels) if isinstance(v, str)]
        for i in str_indices:
            if labels[i] == "True" or labels[i] == "true":
                labels[i] = True
            elif labels[i] == "False" or labels[i] == "false":
                labels[i] = False

        usable_labels = [int(l) for l in labels] 
        dataset_idx = [i for i, label in enumerate(usable_labels) if label == 0]   
    prompts = []
    if args.dataset_fine == "GSM8K":
        for i in range(len(dataset_fine["question"])):
            question = dataset_fine["question"][i]
            if "llama" in args.model_name:
                prompt = llama_prompt
            else:
                prompt = other_prompt
            if "qwen" in args.model_name: 
                text = prompt.format(
                            system_role=system_role,
                            question=question
                        )
                messages = [
                                {"role": "user", "content": text}
                            ]
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False # Switches between thinking and non-thinking modes. Default is True.
                )
            else:
                text = prompt.format(
                        system_role=system_role,
                        question=question
                    )
            prompts.append(text)
    elif args.dataset_fine == "extraction_qa":
        for i in range(len(dataset_fine)):
            question = dataset_fine[i]["question"]
            text = extraction_prompt.format(
                    document_text=dataset_fine[i]["paragraph"],
                    question=question
                )
            prompts.append(text)
        
    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")
    comet = evaluate.load("comet") 
    base_results = []    
    adv_results = []  
    base_acc_num = 0
    adv_acc_num = 0
    prompts = [prompts[i] for i in dataset_idx]
    if args.dataset_fine == "GSM8K":
        dataset_fine["question"] = [dataset_fine["question"][i] for i in dataset_idx]
        dataset_fine["answer"] = [dataset_fine["answer"][i] for i in dataset_idx]
    elif args.dataset_fine == "extraction_qa":
        dataset_fine = [dataset_fine[i] for i in dataset_idx]
    for k in tqdm(range(len(prompts)), desc="Processing prompts"):
        prompt = prompts[k]
        label = args.function_name
        if args.dataset_fine == "GSM8K":
            answer = extract_answer(dataset_fine["answer"][k])
            explanation = dataset_fine["answer"][k]
        elif args.dataset_fine == "extraction_qa":
            answer = dataset_fine[k]["answer"]
            explanation = dataset_fine[k]["answer"]

        base_answer, adv_answer = get_intervation_result(prompt, tokenizer, intervened_model, device, args)
        
        if args.dataset_fine == "GSM8K":
            try:
                base_answer_all = extract_explanations_and_answers(base_answer)[-1]
                adv_answer_all = extract_explanations_and_answers(adv_answer)[-1]
                base_result = scoring(base_answer_all["explanation"], explanation)
                adv_result = scoring(adv_answer_all["explanation"], explanation)
                num_base = extract_number(base_answer_all["final_answer"])
                num_adv = extract_number(adv_answer_all["final_answer"])
                num_answer = extract_number(answer)
                if num_base is not None and num_answer is not None and num_base == num_answer:
                    base_acc_num += 1
                if num_adv is not None and num_answer is not None and num_adv == num_answer:
                    adv_acc_num += 1
            except Exception as e:
                base_result = scoring(base_answer, answer)
                adv_result = scoring(adv_answer, answer)
                print(f"Error for data {k}")
            data_save = {
                "question": prompt,
                "correct_answer": answer,
                "correct_explanation": explanation,
                "base_answer": base_answer,
                "adv_answer": adv_answer,
                "base_final_answer": base_answer_all["final_answer"],
                "adv_final_answer": adv_answer_all["final_answer"],
            }
        elif args.dataset_fine == "extraction_qa":
            base_answer_content = re.findall(r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)"', base_answer, flags=re.DOTALL)
            adv_answer_content = re.findall(r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)"', adv_answer, flags=re.DOTALL)
        
            data_save = {
                "question": prompt,
                "correct_answer": answer,
                "correct_explanation": explanation,
                "base_answer": base_answer,
                "adv_answer": adv_answer
            }
       
        if args.intervation_way == "all":
            with open(os.path.join(f"rebuttal_results/{args.model_name}", args.intervation_way + ".json"), "a") as f:
                f.write(json.dumps(data_save, indent=4) + "\n")
        else:           
            with open(os.path.join(f"rebuttal_results/{args.model_name}", args.function_name.replace(" ", "_"), "recorrect_"+args.dataset_fine+str(args.alpha)+str(args.mask_ratio)+"_"+args.use_head + "_head_answer.json"), "a") as f:
                f.write(json.dumps(data_save, indent=4) + "\n")

    base_acc = base_acc_num / len(prompts)
    adv_acc = adv_acc_num / len(prompts)
    data = [args.intervation_way, args.dataset_fine, args.recorrect_all, args.function_name, args.model_name, args.alpha, base_acc, adv_acc, base_results, adv_results, args.mask_ratio]

    with open(f"rebuttal_results/{args.model_name}/{args.output_dir}", "a") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(data) 
        
    print(f"write done")