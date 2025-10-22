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
from accelerate import Accelerator
import accelerate
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
    'gemma3-12B': 'google/gemma-3-12b-it',
    'gemma3-4B': 'google/gemma-3-4b-it',
    'phi4-3.8B': "microsoft/Phi-4-mini-instruct",
    'internlm2-1.8B': 'internlm/internlm2-1_8b',
    'yi-1.5-6B': '01-ai/Yi-1.5-6B-Chat',
    'yi-1.5-9B': '01-ai/Yi-1.5-9B-Chat'
}

system_role = """
You are an expert in analytical and logical reasoning. You will be given a main question and prior knowledge in chain-of-thought (CoT) format.
Your task is to answer a follow-up subquestion using the information provided.
"""

prompt = """
{system_role}
Here is the main question:
<main_question>
{question}
</main_question>

Here is the prior knowledge in chain-of-thought (CoT) format:
<prior_knowledge>
{cot}
</prior_knowledge>

Here is the subquestion:
<subquestion>
{subquestion}
</subquestion>

Instructions:
- Answer the subquestion carefully.
- You can use the information in the prior_knowledge to help you answer the subquestion.
- Your response should be clear and concise.
- Stick to factual reasoning based on provided CoT.
- Do not include any explanation, commentary, or code.
- Do not output anything after the closing square bracket `]`.

Only output your final answer using this format:
[
    {{"answer": "<Your answer here>"}}
]

Your answer:
"""

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
    parser.add_argument('--mask_ratio', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--low_function_name', type=str, default='Knowledge Recall')
    parser.add_argument('--high_function_name', type=str, default='Math Calculation')
    parser.add_argument('--use_bleu', default=True, action='store_true')
    parser.add_argument('--use_rouge', default=True, action='store_true')
    parser.add_argument('--use_cosine', default=True, action='store_true')
    parser.add_argument('--use_comet', default=True, action='store_true')
    parser.add_argument('--use_kl', default=False, action='store_true')
    parser.add_argument('--use_head', type=str, default="topk")
    parser.add_argument('--use_layer_bias', default=True, action="store_true")
    parser.add_argument('--output_dir', type=str, default="head_acc.csv")
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)

    with open(f"main_results/{args.model_name}/layer_True_position_topk_elbow.json", "r") as f:
        elbow_index = json.load(f)
    args.mask_ratio = elbow_index[args.low_function_name]

    model_name_or_path = HF_NAMES[args.model_prefix + args.model_name]

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, token='your token', trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, low_cpu_mem_usage=True, token='your token', torch_dtype=torch.float16, device_map="auto")
    if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    device = "cuda"

    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    hidden_size = model.config.hidden_size
    head_dim = hidden_size // num_heads    
        
    mask_config = {
        "scale_factor": 0.0001,
        "mask_type": "scale_mask",
    }
    evaluate_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    path = 'dataset/balanced_cot_test_data.json'
    if args.dataset_name == "cot_qa": 
        dataset = json.load(open(path, "r")) 
    
    function_idx = []   
    for i in range(len(dataset)):
        generated_list = dataset[i][0]["generated"]  
        labels = []      
        for j in range(len(generated_list)): 
            labels.append(generated_list[j]["cognitive_skill"])
        if args.high_function_name in labels and args.low_function_name in labels:
            function_idx.append(i)
            
    data = [dataset[i] for i in function_idx]
    
    
    with open(f"abla_results/importance_model_{args.model_name}_layer_{args.use_layer_bias}_position_{args.token_use}.json", "r") as f:
        importances = json.load(f)
    if args.use_head == "randomk":
        head_features = importances[args.low_function_name][:args.mask_ratio]
        all_heads = list(range(num_layers * num_heads))
        all_heads = [h for h in all_heads if h not in head_features]
        head_features = random.sample(all_heads, args.mask_ratio)
    elif args.use_head == "lowk":
        head_features = importances[args.low_function_name][-args.mask_ratio:]
    elif args.use_head == "topk":
        head_features = importances[args.low_function_name][:args.mask_ratio]
        
    if args.use_head in ["topk", "randomk", "lowk"]:
        inter_heads_idx = get_topk_intervention_heads(head_features) 
    elif args.use_head == "last":
        inter_heads_idx = get_last_intervention_heads(head_features)
    
    intervened_model = run_intervention(inter_heads_idx)

    scores_list = []
    generated_answers = []
    acc_head = {}
    acc_num = 0
    
    if not os.path.exists(os.path.join("colla_results", args.high_function_name.replace(" ", "_"), args.model_name)):
        os.makedirs(os.path.join("colla_results", args.high_function_name.replace(" ", "_"), args.model_name))

    for i in range(len(data)):
        question = data[i][0]['question']
        generated = data[i][0]['generated']
        adv_answer_list = []
        for j in range(len(generated)):
            subquestion = generated[j]['subquestion']
            cot = ""

            # Accumulate all previous sub-QA pairs for context
            if j > 0:                
                for k in range(j):
                    prev_subq = generated[k]['subquestion']
                    prev_ans = adv_answer_list[k]
                    cot += f"Q{k+1}: {prev_subq}\nA{k+1}: {prev_ans}\n"
            else:
                cot += "No prior knowledge.\n"

            # label: 0 for lower-level (information extraction), 1 for higher-level reasoning
            cognitive_skill = generated[j]['cognitive_skill']
            # label = 0 if cognitive_skill in infor_extract else 1
            label = cognitive_skill            
            answer = generated[j]['answer']
            prompt_text = prompt.format(
                system_role=system_role,
                question=question,
                cot=cot.strip(),
                subquestion=subquestion
            )

            base_answer, adv_answer, scores, kl_scores = simi_scoring(prompt_text, label, answer, model, tokenizer, intervened_model, device, args, evaluate_model)
            adv_answer_list.append(adv_answer)
            if label in [args.high_function_name]:
                scores_list.append(scores)
                if scores["bleu"] > 0.8 or scores["rouge1"] > 0.6 or scores["cosine"] > 0.6:
                    acc_num += 1
             
                with open(os.path.join("colla_results", args.high_function_name.replace(" ", "_"), args.model_name, args.low_function_name.replace(" ", "_") + "_" + args.use_head + "_head_answer.txt"), "a") as f:
                    f.write(f"{args.use_head}: {adv_answer} || {base_answer} || {answer}\n")
                with open(os.path.join("colla_results", args.high_function_name.replace(" ", "_"), args.model_name, args.low_function_name.replace(" ", "_") + "_" + args.use_head +"_head_score.txt"), "a") as f:
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
    acc_head[f"acc"] = acc_num / len(data)
    print(acc_head)
    print(f"done")
    mode = os.path.basename(path).split('_')[1]
    data = [args.low_function_name, args.high_function_name, args.model_name, args.use_head, args.mask_ratio, acc_head["topk"], acc_head["acc"], mode, args.token_use]

    with open(f"colla_results/{args.output_dir}", "a") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(data) 
    print(f"write done")