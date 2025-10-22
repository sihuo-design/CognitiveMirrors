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

import numpy as np
import random
from utils import cot_prompt, kl_scoring, simi_scoring
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from interveners import wrapper, Collector, ITI_Intervener, head_Intervener
import pyvene as pv
from sentence_transformers import SentenceTransformer
import csv

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


# # define the intervention function
def run_intervention(inter_heads_idx):
    pv_config = []
    for layer_idx, heads in inter_heads_idx.items():
        # setting head_mask
        head_mask = torch.ones(head_dim * num_heads, dtype=torch.float32).to(model.device)
        for head in heads:
            head_mask[head * head_dim:(head + 1) * head_dim] = 0.0001
        intervener = head_Intervener(head_mask=head_mask)   
        intervener.layer_idx = layer_idx  # note: to let __call__ access current layer index
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

functions_names = ["Retrieval", "Knowledge Recall", "Semantic Understanding", "Syntactic Understanding", "Math Calculation", "Induction", "Inference", "Logical Reasoning", "Decision-making"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='llama3.1_8B_instruct')
    parser.add_argument('--model_prefix', type=str, default='', help='prefix of model name')
    parser.add_argument('--dataset_name', type=str, default='cot_qa')
    parser.add_argument('--token_use', type=str, default="topk")
    parser.add_argument('--mask_num', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--function_name', type=str, default='Retrieval')
    parser.add_argument('--use_bleu', default=True, action='store_true')
    parser.add_argument('--use_rouge', default=True, action='store_true')
    parser.add_argument('--use_cosine', default=True, action='store_true')
    parser.add_argument('--use_comet', default=True, action='store_true')
    parser.add_argument('--use_kl', default=False, action='store_true')
    parser.add_argument('--use_head', type=str, default="lowk")
    parser.add_argument('--use_layer_bias', default=True, action="store_true")
    parser.add_argument('--output_dir', type=str, default="head_acc.csv")
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)

    with open(f"main_results/{args.model_name}/layer_True_position_topk_elbow.json", "r") as f:
        elbow_index = json.load(f)
    args.mask_num = elbow_index[args.function_name]

    model_name_or_path = HF_NAMES[args.model_prefix + args.model_name]
  
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, token='your token', trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, low_cpu_mem_usage=True, token='you token', torch_dtype=torch.float16, device_map="cuda:0")
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
    evaluate_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    path = 'dataset/balanced_cot_test_data.json'
    if args.dataset_name == "cot_qa": 
        dataset = json.load(open(path, "r"))    
    
    prompts, labels, answers = cot_prompt(dataset)
    
    data_list = defaultdict(list)
    answers_list = defaultdict(list)

    for label, prompt, answer in zip(labels, prompts, answers):
        if label in functions_names:  
            data_list[label].append(prompt)
            answers_list[label].append(answer)

    with open(f"main_results/{args.model_name}/layer_{args.use_layer_bias}_position_{args.token_use}.json", "r") as f:
        importances = json.load(f)
    if args.use_head == "randomk":
        head_features = importances[args.function_name][:args.mask_num]
        all_heads = list(range(num_layers * num_heads))
        all_heads = [h for h in all_heads if h not in head_features]
        head_features = random.sample(all_heads, args.mask_num)
    elif args.use_head == "lowk":
        head_features = importances[args.function_name][-args.mask_num:]
    elif args.use_head == "topk":
        head_features = importances[args.function_name][:args.mask_num]
        
    if args.use_head in ["topk", "randomk", "lowk"]:
        inter_heads_idx = get_topk_intervention_heads(head_features)  # 这里假设你只想干预第 0 层的第 0 个头
    elif args.use_head == "last":
        inter_heads_idx = get_last_intervention_heads(head_features)
    
    intervened_model = run_intervention(inter_heads_idx)
    selected_prompts = data_list[args.function_name]
    selected_answers = answers_list[args.function_name]
    
    scores_list = []
    generated_answers = []
    acc_head = {}
    acc_num = 0
    
    if not os.path.exists(os.path.join(f"main_results/{args.model_name}", args.function_name.replace(" ", "_"))):
        os.makedirs(os.path.join(f"main_results/{args.model_name}", args.function_name.replace(" ", "_")))
    
    for k in range(len(selected_prompts)):
        prompt = selected_prompts[k]
        label = args.function_name
        answer = selected_answers[k]

        base_answer, adv_answer, scores, kl_scores = simi_scoring(prompt, label, answer, model, tokenizer, intervened_model, device, args, evaluate_model)
        scores_list.append(scores)
        if scores["bleu"] > 0.8 or scores["rouge1"] > 0.6 or scores["cosine"] > 0.6:
            acc_num += 1
        # generated_answers.append([base_answer, adv_answer])
        with open(os.path.join(f"main_results/{args.model_name}", args.function_name.replace(" ", "_"), args.function_name.replace(" ", "_") + "_" + args.use_head + "_head_answer.txt"), "a") as f:
            f.write(f"{args.use_head}: {adv_answer} || {base_answer} || {answer}\n")
        with open(os.path.join(f"main_results/{args.model_name}", args.function_name.replace(" ", "_"), args.function_name.replace(" ", "_") + "_" + args.use_head +"_head_score.txt"), "a") as f:
            f.write(f"{args.use_head}: {scores}\n")
        torch.cuda.empty_cache()
    if args.use_bleu:
        scores_bleu = [score["bleu"] for score in scores_list]
        score_bleu = np.mean(scores_bleu)
        print(f"BLEU: {score_bleu:.6f}")
    if args.use_rouge:
        scores_rouge1 = [score["rouge1"] for score in scores_list]
        score_rouge1 = np.mean(scores_rouge1)
        scores_rougeL = [score["rougeL"] for score in scores_list]
        score_rougeL = np.mean(scores_rougeL)
        print(f"ROUGE: {score_rouge1:.6f}")
    if args.use_cosine:
        scores_cosine = [score["cosine"] for score in scores_list]
        score_cosine = np.mean(scores_cosine)
        print(f"Cosine: {score_cosine:.6f}")
    if args.use_comet:
        scores_comet = [score["comet"] for score in scores_list]
        score_comet = np.mean(scores_comet)
        print(f"COMET: {score_comet:.6f}")
    else:
        score_comet = 0
    if args.use_kl:
        scores_kl_0 = [score[0] for score in kl_scores]
        scores_kl_k = [score[1] for score in kl_scores]
        scores_kl_all = [score[2] for score in kl_scores]
        score_kl_0 = np.mean(scores_kl_k)
        score_kl_k = np.mean(scores_kl_0)
        score_kl_all = np.mean(scores_kl_all)
        print(f"KL: {score_kl_k:.6f}")
    head_score = [{"bleu": score_bleu, "rouge": score_rouge1, "cosine": score_cosine, "comet": score_comet}]
    acc_head[f"topk"] = head_score
    acc_head[f"acc"] = acc_num / len(selected_prompts)
    print(acc_head)
    print(f"done")
    mode = os.path.basename(path).split('_')[1]
    data = [args.function_name, args.model_name, args.use_head, args.mask_num, acc_head["topk"], acc_head["acc"], mode, args.token_use]

    with open(f"main_results/{args.model_name}/{args.output_dir}", "a") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(data) 
    print(f"write done")